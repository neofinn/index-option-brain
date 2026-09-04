"""Live adapter for NSE's public web API.

This is a **live** adapter: every number it returns was observed on the
exchange's own endpoints. Nothing here is simulated, and nothing is filled in
with a plausible default — a value NSE does not publish comes back as `None`,
and a capability NSE does not serve raises `DataAdapterError` rather than
returning something that looks like data.

What NSE public actually serves (verified against the live endpoints)
---------------------------------------------------------------------
* `/api/allIndices` — index last/open/high/low/previousClose for NIFTY 50,
  NIFTY BANK and the rest, **including INDIA VIX** as an index row. One
  request covers both spot and VIX.
* `/api/option-chain-contract-info` — the tradeable expiry dates. Weekly
  expiries are **Tuesdays**; the exchange moved off Thursday.
* `/api/option-chain-v3` — the full chain: LTP, best bid/ask with depth,
  implied volatility, open interest and its change, and traded volume.

What it does not serve
----------------------
* **No greeks.** Only `impliedVolatility`. Delta, gamma, theta and vega are
  computed here from the live premium and IV via `analytics.pricing`, which is
  why that module is production code rather than a test helper. They are
  computed against the **forward solved from put-call parity**, not against
  spot: NIFTY's forward is set by the futures and stood 43.7 points above
  spot on 3 Sep 2026 against a carry of 22.4. Pricing off spot pushes that
  gap into the volatility — the same strike solved to a 10.6% call IV and an
  8.7% put IV — and biases every delta derived from it.
* **No historical bars.** `/api/historical/indicesHistory` answers an
  automated client with an anti-bot HTML interstitial. `get_index_bars`
  raises instead of inventing candles; bars come from a broker adapter, or
  from aggregating live snapshots forward.
* **No live constituent quotes.** `/api/equity-stockIndices` returns 404 for
  this client, so intraday breadth is unavailable here. It is not entirely
  unavailable: `/api/market-data-pre-open` serves all 50 constituents during
  the opening auction, which `nse_preopen.NsePreOpenAdapter` reads. That
  covers 09:00-09:15 and goes stale afterwards; continuous breadth still
  needs a broker adapter.
* **No account or order placement.** It is a data source, not a broker.

Operational notes
-----------------
* The API rejects requests that do not carry the cookies its HTML pages set,
  so the session is warmed up against the option-chain page first, and
  re-warmed once if a response comes back rejected or non-JSON.
* Timestamps are IST-local strings with no zone; they are converted to
  timezone-aware UTC here so nothing downstream has to guess.
* Greeks are computed as of the **payload's own timestamp**, not wall clock.
  Using `now()` would make a replayed or delayed snapshot produce greeks for a
  moment that snapshot never described.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta, timezone
from decimal import Decimal
from itertools import pairwise
from typing import TYPE_CHECKING, Any, Self

from index_option_brain.analytics.pricing import (
    DEFAULT_RISK_FREE_RATE,
    ForwardEstimate,
    forward_from_parity,
    greeks_from_iv,
    implied_volatility,
)
from index_option_brain.contracts.enums import BarInterval, OptionType
from index_option_brain.contracts.instruments import (
    Bar,
    Greeks,
    IndexQuote,
    IndexSpec,
    OptionContractSpec,
    OptionQuote,
)
from index_option_brain.contracts.provider import (
    AuthMethod,
    Capability,
    ProviderDescriptor,
    ProviderKind,
)
from index_option_brain.data.adapters.base import (
    DataAdapterError,
    IndexDataAdapter,
    OptionsChainAdapter,
    VolatilityDataAdapter,
)
from index_option_brain.data.http import HttpError, HttpSession, HttpxSession

if TYPE_CHECKING:  # pragma: no cover
    from index_option_brain.data.dhan_instruments import DhanInstrumentMaster

NSE_BASE = "https://www.nseindia.com"
_WARMUP_URL = f"{NSE_BASE}/option-chain"
_ALL_INDICES_URL = f"{NSE_BASE}/api/allIndices"
_CONTRACT_INFO_URL = f"{NSE_BASE}/api/option-chain-contract-info"
_OPTION_CHAIN_URL = f"{NSE_BASE}/api/option-chain-v3"

IST = timezone(timedelta(hours=5, minutes=30), name="IST")

# Indian equity derivatives expire at the close of the session, and the option
# is worth its intrinsic value from that moment. Time-to-expiry is measured to
# this instant, not to midnight, because a same-day weekly priced to midnight
# would carry eight and a half hours of value it does not have.
MARKET_CLOSE_IST = time(15, 30)

# A browser User-Agent is required: the endpoint returns an interstitial to
# clients it does not recognize. This is a compatibility requirement of the
# public API, not an attempt to hide what the client is.
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
_API_HEADERS = {"Accept": "*/*", "Referer": _WARMUP_URL}
_PAGE_HEADERS = {"Accept": "text/html,application/xhtml+xml"}

NSE_INDIA_VIX_NAME = "INDIA VIX"

# Bounds on a believable single-strike implied volatility, in percentage
# points. Loose on purpose: they exist to reject arithmetic artifacts, not
# to express a view on how volatile the index can get.
MIN_PLAUSIBLE_IV_PERCENT = Decimal("0.5")
MAX_PLAUSIBLE_IV_PERCENT = Decimal(300)


@dataclass(frozen=True)
class NseIndexConfig:
    """Contract specifications for one index.

    These are exchange contract specifications published by circular, not
    market observations, and NSE's own public endpoints do not expose them —
    the derivative-quote endpoint carrying `marketLot` returns 404 to this
    client.

    **Do not rely on the defaults below.** Build this table from Dhan's public
    instrument master instead, via `index_config_from_master`: it is
    authoritative, needs no credentials, and carries the lot size per expiry
    so a revision in flight is handled correctly.

    The reason for that instruction is a bug these defaults actually had. The
    NIFTY lot size here was 75, and the exchange's own record says 65 — on
    every listed expiry. Every position size, max loss, margin estimate and
    exposure figure derived from it was about 15% overstated, and the
    Execution Gate's LOT_SIZE_VALID check could not catch it, because it
    compares a leg's lot size against `IndexSpec.lot_size` and both came from
    this one wrong constant. A consistency check between two copies of a
    number is not a correctness check.
    """

    nse_index_name: str
    """The name NSE uses in `/api/allIndices` — "NIFTY 50", not "NIFTY"."""
    display_name: str
    lot_size: int
    strike_step: Decimal
    tick_size: Decimal = Decimal("0.05")


# A last-resort fallback for running with no network access to the instrument
# master. The lot sizes here were verified against Dhan's master on
# 02-Sep-2026; they are a snapshot and they will go stale. Prefer
# `index_config_from_master`.
DEFAULT_INDEX_CONFIG: dict[str, NseIndexConfig] = {
    "NIFTY": NseIndexConfig(
        nse_index_name="NIFTY 50",
        display_name="Nifty 50",
        lot_size=65,
        strike_step=Decimal(50),
    ),
    "BANKNIFTY": NseIndexConfig(
        nse_index_name="NIFTY BANK",
        display_name="Nifty Bank",
        lot_size=30,
        strike_step=Decimal(100),
    ),
}

# The NSE names for the indices this system can be pointed at, so a config
# built from the instrument master knows what to call them on `/api/allIndices`.
NSE_INDEX_NAMES: dict[str, tuple[str, str]] = {
    "NIFTY": ("NIFTY 50", "Nifty 50"),
    "BANKNIFTY": ("NIFTY BANK", "Nifty Bank"),
    "FINNIFTY": ("NIFTY FINANCIAL SERVICES", "Nifty Financial Services"),
    "MIDCPNIFTY": ("NIFTY MIDCAP SELECT", "Nifty Midcap Select"),
}


def index_config_from_master(
    master: DhanInstrumentMaster,
    *,
    symbols: tuple[str, ...] = ("NIFTY", "BANKNIFTY"),
) -> dict[str, NseIndexConfig]:
    """Build the contract table from the exchange's own record.

    This is the supported way to configure the adapter. Lot size, tick size
    and strike step all come from Dhan's public instrument master rather than
    from a constant, which is what stops a circular revision from silently
    mis-sizing every order.

    A symbol the master does not list is skipped rather than defaulted: an
    index whose contract size cannot be verified must not enter the sizing
    path at all.
    """
    config: dict[str, NseIndexConfig] = {}
    for symbol in symbols:
        names = NSE_INDEX_NAMES.get(symbol.upper())
        if names is None:
            continue
        try:
            expiries = master.expiries(symbol)
            near = expiries[0] if expiries else None
            lot_size = master.lot_size(symbol, near)
            tick_size = master.tick_size(symbol)
        except DataAdapterError:
            continue
        step = master.strike_step(symbol, near) if near is not None else None
        config[symbol.upper()] = NseIndexConfig(
            nse_index_name=names[0],
            display_name=names[1],
            lot_size=lot_size,
            strike_step=step if step is not None else Decimal(50),
            tick_size=tick_size,
        )
    return config

NSE_PUBLIC_DESCRIPTOR = ProviderDescriptor(
    provider_id="nse_public",
    display_name="NSE India (public web API)",
    kind=ProviderKind.DATA,
    auth=AuthMethod.NONE,
    capabilities=frozenset(
        {
            Capability.INDEX_QUOTE,
            Capability.EXPIRY_LIST,
            Capability.OPTION_CHAIN,
            Capability.INDIA_VIX,
        }
    ),
    implemented=True,
    # Every capability was probed against the live endpoint, and the
    # parsing is pinned against recorded responses.
    verified=True,
    docs_url="https://www.nseindia.com/option-chain",
    notes=(
        (
            "No greeks published — delta/gamma/theta/vega are computed from "
            "the live premium and IV."
        ),
        (
            "No historical bars: the history endpoint blocks automated "
            "clients. Pair with a broker adapter, or aggregate live snapshots."
        ),
        "No constituent quotes, so index breadth needs another provider.",
        "Data only: cannot hold an account or place an order.",
        (
            "Unauthenticated and rate-limited. Snapshots are cached briefly "
            "so one analysis cycle makes one request."
        ),
    ),
)


def _book_mid(leg: dict[str, Any]) -> float | None:
    """Mid of a two-sided book, or None when either side is missing.

    None, never the last traded price: parity needs both legs priced at the
    same instant, and a last trade is priced at whenever it happened.
    """
    bid = _optional_decimal(leg.get("buyPrice1"))
    ask = _optional_decimal(leg.get("sellPrice1"))
    if bid is None or ask is None or bid <= 0 or ask <= 0 or ask < bid:
        return None
    return float((bid + ask) / 2)


def _to_decimal(value: Any, field: str) -> Decimal:
    """Convert a JSON number to Decimal via its string form.

    Going through `str` matters: `Decimal(23914.45)` captures the binary
    float's error, while `Decimal("23914.45")` is the number NSE sent.
    """
    if value is None:
        raise DataAdapterError(f"NSE payload is missing required field {field!r}")
    try:
        return Decimal(str(value))
    except (ArithmeticError, TypeError, ValueError) as exc:
        raise DataAdapterError(
            f"NSE field {field!r} is not a number: {value!r}"
        ) from exc


def _optional_decimal(value: Any) -> Decimal | None:
    """A price NSE reports as 0 or null means "no quote on this side".

    Returning `None` keeps that distinct from a genuine price of zero, which is
    what lets `OptionQuote.mid` fall back to LTP instead of computing a mid
    against a side that was never quoted.
    """
    if value is None:
        return None
    try:
        parsed = Decimal(str(value))
    except (ArithmeticError, TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _to_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(Decimal(str(value)))
    except (ArithmeticError, TypeError, ValueError):
        return 0


def parse_ist_timestamp(raw: str) -> datetime:
    """Parse an NSE timestamp string into timezone-aware UTC.

    NSE sends IST-local strings with no zone and inconsistent precision:
    `/api/allIndices` sends "02-Sep-2026 15:30" while the chain sends
    "02-Sep-2026 15:40:00". Both are handled; anything else raises rather than
    silently becoming `now()`, because a wrong timestamp on a snapshot
    corrupts every time-to-expiry computed from it.
    """
    text = raw.strip()
    for fmt in ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y %H:%M", "%d-%b-%Y"):
        try:
            naive = datetime.strptime(text, fmt)  # noqa: DTZ007 - IST applied below
        except ValueError:
            continue
        return naive.replace(tzinfo=IST).astimezone(UTC)
    raise DataAdapterError(f"Unrecognized NSE timestamp format: {raw!r}")


def parse_nse_date(raw: str) -> date:
    """Parse an NSE date, accepting both formats it uses.

    Expiry appears as "08-Sep-2026" in the expiry list but as "08-09-2026"
    inside a chain leg, in the same response family.
    """
    text = raw.strip()
    for fmt in ("%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date()  # noqa: DTZ007 - date only
        except ValueError:
            continue
    raise DataAdapterError(f"Unrecognized NSE date format: {raw!r}")


def expiry_instant(expiry: date) -> datetime:
    """The UTC instant an expiry settles: 15:30 IST on the expiry date."""
    return datetime.combine(expiry, MARKET_CLOSE_IST, tzinfo=IST).astimezone(UTC)


def years_to_expiry(*, as_of: datetime, expiry: date) -> float:
    """Calendar years from `as_of` to settlement, floored at zero.

    Calendar rather than trading time: premium decays across the weekend, and
    a Tuesday-expiry weekly held from Friday loses three days of value.
    """
    seconds = (expiry_instant(expiry) - as_of).total_seconds()
    return max(0.0, seconds / (365.0 * 24 * 3600))


def infer_strike_step(strikes: list[Decimal]) -> Decimal | None:
    """The most common gap between adjacent strikes, or None.

    Derived rather than assumed because NSE's strike list mixes a dense band
    around spot with sparse legacy strikes far away, so the smallest gap and
    the average gap are both wrong. The mode is the step that is actually
    tradeable.
    """
    ordered = sorted(set(strikes))
    if len(ordered) < 3:
        return None
    gaps = Counter(
        second - first for first, second in pairwise(ordered) if second > first
    )
    if not gaps:
        return None
    return gaps.most_common(1)[0][0]


class NsePublicAdapter(IndexDataAdapter, OptionsChainAdapter, VolatilityDataAdapter):
    """Live index, option chain and India VIX from NSE's public API."""

    descriptor = NSE_PUBLIC_DESCRIPTOR

    def __init__(
        self,
        session: HttpSession | None = None,
        *,
        index_config: dict[str, NseIndexConfig] | None = None,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
        snapshot_ttl_seconds: float = 5.0,
        compute_greeks: bool = True,
        prefer_published_iv: bool = False,
        max_relative_spread_for_iv: float = 0.15,
    ) -> None:
        """
        `snapshot_ttl_seconds` caches the all-indices payload briefly. The
        point is **coherence**, not just politeness to a free endpoint: a
        single analysis cycle reads the index spot and India VIX separately,
        and if those two reads landed on different snapshots the resulting
        MarketState would describe a market that never existed. It is the same
        idempotence property the simulator guarantees.

        `prefer_published_iv` chooses whose implied volatility to trust. NSE
        computes its published IV from the **last traded price**, which on a
        thin strike can be minutes stale: measured live, the 22,900 CE marked
        965.20/1,082.25 on the book while NSE published 46.55% IV off an LTP
        of 1,190 — a number no live quote supports. Marking to the mid of a
        two-sided book is what an options desk does, so by default the IV is
        re-derived from the mid and the published figure is used only when
        there is no markable book. Set this True to pass NSE's number through
        unchanged.

        `max_relative_spread_for_iv` is the width past which a mid stops
        meaning anything. Beyond it the strike gets no IV and no greeks, which
        excludes it from strike ranking — the correct outcome for a market too
        wide to trade, and better than ranking it on a fabricated delta.
        """
        self._owns_session = session is None
        self._session: HttpSession = session or HttpxSession(headers=_BROWSER_HEADERS)
        self._index_config = dict(index_config or DEFAULT_INDEX_CONFIG)
        self._rate = risk_free_rate
        self._snapshot_ttl = snapshot_ttl_seconds
        self._compute_greeks = compute_greeks
        self._prefer_published_iv = prefer_published_iv
        self._max_spread_for_iv = Decimal(str(max_relative_spread_for_iv))
        self._warmed_up = False
        self._last_forward: dict[tuple[str, date], ForwardEstimate | None] = {}
        self._lock = asyncio.Lock()
        self._indices_cache: tuple[float, dict[str, Any]] | None = None

    # ------------------------------------------------------------------ HTTP

    async def aclose(self) -> None:
        if self._owns_session:
            await self._session.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def _warm_up(self) -> None:
        try:
            await self._session.get(_WARMUP_URL, headers=_PAGE_HEADERS)
        except HttpError as exc:
            raise DataAdapterError(f"Cannot reach NSE: {exc}") from exc
        self._warmed_up = True

    async def _get_json(
        self, url: str, params: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """Fetch and parse JSON, warming up cookies and retrying once.

        The retry covers the one failure mode this endpoint has in normal
        operation: an expired cookie, which comes back either as a rejection
        status or as an HTML interstitial with a 200. Both are retried exactly
        once — a second failure is reported, not papered over, because
        withholding trades on a broken feed is the required behaviour
        (spec §29).
        """
        if not self._warmed_up:
            await self._warm_up()

        for attempt in (1, 2):
            try:
                response = await self._session.get(
                    url, params=params, headers=_API_HEADERS
                )
            except HttpError as exc:
                raise DataAdapterError(f"NSE request failed: {exc}") from exc

            if response.is_ok:
                try:
                    payload = response.json()
                except ValueError:
                    # A 200 carrying HTML is the anti-bot interstitial.
                    if attempt == 1:
                        await self._warm_up()
                        continue
                    raise DataAdapterError(
                        f"NSE returned a non-JSON body for {url} — the request "
                        "was blocked rather than answered"
                    ) from None
                if not isinstance(payload, dict):
                    raise DataAdapterError(
                        f"NSE returned {type(payload).__name__}, expected an "
                        f"object, for {url}"
                    )
                return payload

            if attempt == 1 and response.status_code in (401, 403, 429, 503):
                await self._warm_up()
                continue

            raise DataAdapterError(
                f"NSE returned HTTP {response.status_code} for {url}"
            )

        raise DataAdapterError(f"NSE request to {url} could not be completed")

    async def _all_indices(self) -> dict[str, Any]:
        """The all-indices payload, cached for `snapshot_ttl_seconds`."""
        async with self._lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            cached = self._indices_cache
            if cached is not None and now - cached[0] < self._snapshot_ttl:
                return cached[1]
            payload = await self._get_json(_ALL_INDICES_URL)
            self._indices_cache = (now, payload)
            return payload

    def _index_row(self, payload: dict[str, Any], nse_name: str) -> dict[str, Any]:
        rows = payload.get("data")
        if not isinstance(rows, list):
            raise DataAdapterError("NSE all-indices payload has no 'data' list")
        for row in rows:
            if isinstance(row, dict) and row.get("index") == nse_name:
                return row
        raise DataAdapterError(
            f"NSE all-indices payload does not contain {nse_name!r}"
        )

    def _config(self, symbol: str) -> NseIndexConfig:
        try:
            return self._index_config[symbol.upper()]
        except KeyError:
            raise DataAdapterError(
                f"{symbol!r} is not configured for the NSE public adapter. "
                f"Configured: {sorted(self._index_config)}. Add it via "
                "index_config with a lot size verified against the current "
                "NSE contract specification."
            ) from None

    # ---------------------------------------------------------- index data

    async def get_index_spec(self, symbol: str) -> IndexSpec:
        config = self._config(symbol)
        return IndexSpec(
            symbol=symbol.upper(),
            name=config.display_name,
            lot_size=config.lot_size,
            tick_size=config.tick_size,
            strike_step=config.strike_step,
        )

    async def get_index_quote(self, symbol: str) -> IndexQuote:
        config = self._config(symbol)
        payload = await self._all_indices()
        row = self._index_row(payload, config.nse_index_name)
        timestamp_raw = payload.get("timestamp")
        if not isinstance(timestamp_raw, str):
            raise DataAdapterError("NSE all-indices payload has no timestamp")
        return IndexQuote(
            symbol=symbol.upper(),
            timestamp=parse_ist_timestamp(timestamp_raw),
            ltp=_to_decimal(row.get("last"), "last"),
            open=_to_decimal(row.get("open"), "open"),
            high=_to_decimal(row.get("high"), "high"),
            low=_to_decimal(row.get("low"), "low"),
            previous_close=_to_decimal(row.get("previousClose"), "previousClose"),
            # NSE does not publish an index VWAP on this endpoint. None means
            # "not measured"; a fabricated VWAP would corrupt the VWAP
            # relationship the Index brain reads.
            vwap=None,
        )

    async def get_index_bars(
        self, symbol: str, interval: BarInterval, count: int
    ) -> list[Bar]:
        """Not available from NSE public — raises `DataAdapterError`.

        `/api/historical/indicesHistory` answers an automated client with an
        anti-bot HTML interstitial rather than data. Synthesising bars from
        the current snapshot would produce candles the market never printed
        and would silently corrupt ATR, RSI and every structural level
        derived from them, so this fails loudly instead.
        """
        raise DataAdapterError(
            f"NSE public serves no historical bars ({symbol} {interval}, "
            f"{count} requested): the history endpoint blocks automated "
            "clients. Supply bars from a broker adapter, or aggregate live "
            "snapshots forward with LiveBarAggregator."
        )

    # ------------------------------------------------------------ India VIX

    async def get_india_vix(self) -> tuple[float, float]:
        payload = await self._all_indices()
        row = self._index_row(payload, NSE_INDIA_VIX_NAME)
        current = _to_decimal(row.get("last"), "last")
        previous = _to_decimal(row.get("previousClose"), "previousClose")
        return float(current), float(previous)

    async def get_india_vix_range(self) -> tuple[float, float] | None:
        """The 52-week range, from the same cached snapshot as the level.

        No extra request: `yearHigh` and `yearLow` ride along on the
        all-indices payload that already supplies the spot and the VIX. A
        missing or degenerate range returns None rather than a fabricated
        band, since a zero-width range would put every reading at the same
        percentile.
        """
        payload = await self._all_indices()
        row = self._index_row(payload, NSE_INDIA_VIX_NAME)
        high = _optional_decimal(row.get("yearHigh"))
        low = _optional_decimal(row.get("yearLow"))
        if high is None or low is None or high <= low:
            return None
        return float(high), float(low)

    # ---------------------------------------------------------- option data

    async def get_available_expiries(self, underlying_symbol: str) -> list[date]:
        config = self._config(underlying_symbol)
        payload = await self._get_json(
            _CONTRACT_INFO_URL, {"symbol": underlying_symbol.upper()}
        )
        raw = payload.get("expiryDates")
        if not isinstance(raw, list) or not raw:
            raise DataAdapterError(
                f"NSE published no expiry dates for {config.nse_index_name}"
            )
        # Sorted, because the brains treat the first entry as the near expiry
        # and the endpoint's ordering is not a documented guarantee.
        return sorted({parse_nse_date(str(entry)) for entry in raw})

    async def get_option_chain(
        self, underlying_symbol: str, expiry: date
    ) -> list[OptionQuote]:
        symbol = underlying_symbol.upper()
        config = self._config(symbol)
        payload = await self._get_json(
            _OPTION_CHAIN_URL,
            {
                "type": "Indices",
                "symbol": symbol,
                # NSE's chain endpoint accepts only the "08-Sep-2026" form.
                "expiry": expiry.strftime("%d-%b-%Y"),
            },
        )
        records = payload.get("records")
        if not isinstance(records, dict):
            raise DataAdapterError("NSE chain payload has no 'records' object")
        rows = records.get("data")
        if not isinstance(rows, list) or not rows:
            raise DataAdapterError(
                f"NSE published an empty chain for {symbol} {expiry.isoformat()}"
            )

        timestamp_raw = records.get("timestamp")
        if not isinstance(timestamp_raw, str):
            raise DataAdapterError("NSE chain payload has no timestamp")
        as_of = parse_ist_timestamp(timestamp_raw)
        spot = _to_decimal(records.get("underlyingValue"), "underlyingValue")
        years = years_to_expiry(as_of=as_of, expiry=expiry)

        # Solve the market's forward before pricing anything off it. NIFTY's
        # forward is set by the futures and is not spot * exp(rate * years):
        # pricing off spot alone pushes the difference into the volatility,
        # making a call and a put at the same strike solve to different IVs
        # and biasing every delta computed from them.
        forward = self._forward(rows=rows, spot=spot, years=years)
        self._last_forward[(symbol, expiry)] = forward
        carry = forward.dividend_yield if forward is not None else 0.0

        quotes: list[OptionQuote] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            strike_raw = row.get("strikePrice")
            if strike_raw is None:
                continue
            strike = _to_decimal(strike_raw, "strikePrice")
            for key, option_type in (("CE", OptionType.CE), ("PE", OptionType.PE)):
                leg = row.get(key)
                # A strike can be listed with only one side quoted. Skipping
                # the absent side is correct; inventing it is not.
                if not isinstance(leg, dict):
                    continue
                quotes.append(
                    self._build_quote(
                        leg=leg,
                        symbol=symbol,
                        config=config,
                        strike=strike,
                        option_type=option_type,
                        expiry=expiry,
                        as_of=as_of,
                        spot=spot,
                        years=years,
                        dividend_yield=carry,
                    )
                )

        if not quotes:
            raise DataAdapterError(
                f"NSE chain for {symbol} {expiry.isoformat()} contained no "
                "quotable strikes"
            )
        return quotes

    def _build_quote(
        self,
        *,
        leg: dict[str, Any],
        symbol: str,
        config: NseIndexConfig,
        strike: Decimal,
        option_type: OptionType,
        expiry: date,
        as_of: datetime,
        spot: Decimal,
        years: float,
        dividend_yield: float = 0.0,
    ) -> OptionQuote:
        contract = OptionContractSpec(
            underlying_symbol=symbol,
            expiry=expiry,
            strike=strike,
            option_type=option_type,
            lot_size=config.lot_size,
            tick_size=config.tick_size,
        )
        ltp = _to_decimal(leg.get("lastPrice"), "lastPrice")
        bid = _optional_decimal(leg.get("buyPrice1"))
        ask = _optional_decimal(leg.get("sellPrice1"))

        iv = self._resolve_iv(
            leg=leg,
            spot=spot,
            strike=strike,
            years=years,
            option_type=option_type,
            bid=bid,
            ask=ask,
            dividend_yield=dividend_yield,
        )

        greeks: Greeks | None = None
        if self._compute_greeks and iv is not None and years > 0:
            computed = greeks_from_iv(
                spot=float(spot),
                strike=float(strike),
                years=years,
                iv_percent=float(iv),
                option_type=option_type,
                rate=self._rate,
                dividend_yield=dividend_yield,
            )
            greeks = Greeks(
                delta=Decimal(str(round(computed.delta, 6))),
                gamma=Decimal(str(round(computed.gamma, 10))),
                theta=Decimal(str(round(computed.theta, 4))),
                vega=Decimal(str(round(computed.vega, 4))),
            )

        return OptionQuote(
            contract=contract,
            timestamp=as_of,
            ltp=ltp,
            bid=bid,
            ask=ask,
            volume=_to_int(leg.get("totalTradedVolume")),
            # NSE reports open interest in contracts (lots), not in units of
            # the underlying. Every OI figure in the system is therefore in
            # lots, consistently — the ratios and wall detection the Options
            # brain computes are unit-free, but a mixed convention would break
            # them.
            open_interest=_to_int(leg.get("openInterest")),
            open_interest_change=_to_int(leg.get("changeinOpenInterest")),
            implied_volatility=iv,
            greeks=greeks,
        )

    def _forward(
        self,
        *,
        rows: list[Any],
        spot: Decimal,
        years: float,
    ) -> ForwardEstimate | None:
        """The expiry's forward from put-call parity, or None.

        Only strikes with a two-sided book on *both* legs are offered to the
        solver. A parity difference computed from a last-traded price is a
        difference between two moments, not between two prices, and it is
        exactly the stale-print problem that made NSE's published IV unusable.
        """
        pairs: list[tuple[float, float, float]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            strike_raw = row.get("strikePrice")
            call, put = row.get("CE"), row.get("PE")
            if strike_raw is None or not isinstance(call, dict) or not isinstance(put, dict):
                continue
            call_mid = _book_mid(call)
            put_mid = _book_mid(put)
            if call_mid is None or put_mid is None:
                continue
            pairs.append((float(strike_raw), call_mid, put_mid))

        return forward_from_parity(
            pairs=pairs, spot=float(spot), years=years, rate=self._rate
        )

    def last_forward(self, symbol: str, expiry: date) -> ForwardEstimate | None:
        """The forward solved on the most recent chain fetch, if any.

        Exposed so the builder can put the basis into MarketState without
        re-solving it, and so an operator can see the number the greeks were
        computed against.
        """
        return self._last_forward.get((symbol.upper(), expiry))

    def _resolve_iv(
        self,
        *,
        leg: dict[str, Any],
        spot: Decimal,
        strike: Decimal,
        years: float,
        option_type: OptionType,
        bid: Decimal | None,
        ask: Decimal | None,
        dividend_yield: float = 0.0,
    ) -> Decimal | None:
        """The strike's implied volatility in percentage points, or None.

        Three things can go wrong with NSE's IV, and all three were observed
        on one live snapshot:

        * It is **absent**: NSE sends `impliedVolatility: 0` for strikes it
          has not computed, including strikes with a perfectly tight book (the
          23,600 CE marked 344.25/345.45 and carried no published IV). Zero is
          a missing measurement, not a volatility of zero.
        * It is **stale**: it is computed from the last traded price, so a
          strike that last traded away from the current book carries an IV the
          book does not support — the 22,900 CE published 46.55% off a trade
          at 1,190 while the book stood at 965.20/1,082.25.
        * It is **unrecoverable**: on the widest strikes the mid sits below the
          European lower bound, so no volatility explains it at all.

        The policy in one sentence: mark to the mid of a book tight enough to
        mean something; if there is no such book, use what NSE published;
        otherwise report nothing.

        Two parts of that are deliberate. A failed inversion on a markable
        book does **not** fall back to the published figure — if the live mid
        cannot be explained, a number derived from an older trade is not
        better information, it is older information. And a surviving value
        must be a plausible index volatility; anything outside the bounds is
        discarded rather than clamped, because a clamped IV looks like a
        measurement and a `None` does not.
        """
        published = self._plausible_iv(_optional_decimal(leg.get("impliedVolatility")))
        if self._prefer_published_iv and published is not None:
            return published

        markable = self._markable_price(bid=bid, ask=ask)
        if markable is None or years <= 0:
            return published

        derived = implied_volatility(
            market_price=float(markable),
            spot=float(spot),
            strike=float(strike),
            years=years,
            option_type=option_type,
            rate=self._rate,
            dividend_yield=dividend_yield,
        )
        if derived is None:
            return None
        return self._plausible_iv(Decimal(str(round(derived * 100.0, 2))))

    def _markable_price(
        self, *, bid: Decimal | None, ask: Decimal | None
    ) -> Decimal | None:
        """The mid, when the book is two-sided and tight enough to mean it."""
        if bid is None or ask is None or ask < bid:
            return None
        mid = (bid + ask) / 2
        if mid <= 0:
            return None
        if (ask - bid) / mid > self._max_spread_for_iv:
            return None
        return mid

    def _plausible_iv(self, iv: Decimal | None) -> Decimal | None:
        """Discard an IV outside the range an index option can occupy.

        India VIX has ranged roughly 8-90 over its history; single strikes go
        wider on a smile, so the bounds here are deliberately loose. They only
        catch numbers that are arithmetic artifacts rather than volatility.
        """
        if iv is None:
            return None
        if iv < MIN_PLAUSIBLE_IV_PERCENT or iv > MAX_PLAUSIBLE_IV_PERCENT:
            return None
        return iv
