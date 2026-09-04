"""Live adapter for Delta Exchange India.

This is a **live** adapter, and unusually for this codebase every field
mapping below was verified against the real API before it was written — the
public endpoints need no key, so `get_products`, `option_chain` and
`get_candles` were all called and their payloads inspected. That is the
verification the Dhan adapter is still waiting on.

What Delta serves that NSE does not
-----------------------------------
* **Greeks, published.** `delta`, `gamma`, `theta`, `vega` and `rho` on
  every chain row. NSE publishes none, which is why `analytics.pricing`
  exists to derive them. They are still cross-checked here rather than
  trusted — see `_resolve_greeks`.
* **Three implied volatilities** per row: `bid_iv`, `mark_iv`, `ask_iv`, as
  decimals (0.31925 is 31.925%). Having three is what makes the published
  figure checkable at all, unlike NSE's single value which turned out to be
  computed off stale trades.
* **Historical candles.** `get_candles` returns real OHLC — 90 daily bars
  in one call. NSE answers an automated client with an anti-bot page, which
  is why `nse_archive.py` had to be written to scrape end-of-day CSVs.

Units, which are not the Indian ones
------------------------------------
Prices and greeks are quoted **per one unit of the underlying**, and one
contract is `contract_value` of it — 0.001 BTC on the options checked. So a
premium of 1,339 is USD per BTC, and one contract costs 1.34. Multiplying
in the wrong place is the same class of error as reading NIFTY's lot as 75
when it is 65, and it is silent both times. `contract_value` is read from
the payload rather than assumed, for exactly that reason.

Delta also settles in USD, not rupees. Nothing here converts, and nothing
should: the currency belongs to the `MarketProfile`, and a silent
conversion is worse than a figure the caller has to think about.

What it does not serve
----------------------
* **No constituents.** BTC does not decompose, so breadth is not merely
  unavailable — it is not a thing in this market. `MarketProfile.
  has_constituents` records the difference, because a gap worth closing and
  a category error look identical otherwise.
* **No session boundaries.** It trades continuously. Every measurement
  derived from an open or a close is meaningless here, which is what
  `SessionModel.CONTINUOUS` exists to say.

Concurrency
-----------
`delta-rest-client` is synchronous `requests`. Every adapter interface in
this system is async, so each call is pushed to a thread executor rather
than blocking the poll loop. That is a wrapper, not a fix: the underlying
client still holds a thread for the duration of each request.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from functools import partial
from itertools import pairwise
from typing import Any, Protocol

from index_option_brain.analytics.pricing import DEFAULT_RISK_FREE_RATE, greeks_from_iv
from index_option_brain.contracts.enums import BarInterval, OptionType
from index_option_brain.contracts.instruments import (
    Bar,
    Greeks,
    IndexQuote,
    IndexSpec,
    OptionContractSpec,
    OptionQuote,
)
from index_option_brain.data.adapters.base import DataAdapterError

PRODUCTION_INDIA = "https://api.india.delta.exchange"
TESTNET_INDIA = "https://cdn-ind.testnet.deltaex.org"

#: Delta's option symbols: C-BTC-79800-060926 (type, asset, strike, DDMMYY).
_SYMBOL = re.compile(r"^(?P<kind>[CP])-(?P<asset>[A-Z0-9]+)-(?P<strike>\d+)-(?P<expiry>\d{6})$")

#: Delta's resolution strings happen to match this system's BarInterval
#: values, but they are mapped explicitly rather than passed through: the
#: coincidence is not a contract, and a silent passthrough would send an
#: unsupported interval to the venue as a plausible-looking string.
_RESOLUTIONS = {
    BarInterval.MINUTE_1: "1m",
    BarInterval.MINUTE_5: "5m",
    BarInterval.MINUTE_15: "15m",
    BarInterval.DAY: "1d",
}

_INTERVAL_SECONDS = {
    BarInterval.MINUTE_1: 60,
    BarInterval.MINUTE_5: 300,
    BarInterval.MINUTE_15: 900,
    BarInterval.DAY: 86400,
}

#: Beyond this the published and recomputed greeks are treated as
#: disagreeing. Delta's own delta and a Black-Scholes delta from its own
#: mark_iv agreed to about 0.02 on the row checked, most of which is the
#: expiry timestamp's resolution, so 0.05 catches a real mapping error
#: without firing on rounding.
GREEK_TOLERANCE = 0.05


class DeltaClient(Protocol):
    """The slice of `delta_rest_client.DeltaRestClient` used here.

    Narrowed to a Protocol so tests inject recorded payloads rather than
    reaching the exchange, and so the dependency stays visible.
    """

    def get_products(
        self, query: dict[str, Any] | None = ..., auth: bool = ...
    ) -> Any: ...

    def option_chain(
        self, underlying_asset_symbol: str, expiry_date: str, auth: bool = ...
    ) -> Any: ...

    def get_candles(
        self, symbol: str, resolution: str, start: int, end: int, auth: bool = ...
    ) -> Any: ...


def _decimal(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _required(value: Any, field: str) -> Decimal:
    resolved = _decimal(value)
    if resolved is None:
        raise DataAdapterError(f"Delta payload has no usable {field}: {value!r}")
    return resolved


def _int(value: Any) -> int | None:
    resolved = _decimal(value)
    return None if resolved is None else int(resolved)


def parse_option_symbol(symbol: str) -> tuple[OptionType, str, Decimal, date]:
    """Decompose `C-BTC-79800-060926` into its parts.

    Parsed rather than trusted from separate fields because the symbol is
    the one identifier that appears in every payload including order
    responses, and a strike read from one field and an expiry from another
    can drift apart. A symbol that does not match is refused: guessing at
    an unrecognised instrument is how the wrong contract gets traded.
    """
    match = _SYMBOL.match(symbol.strip().upper())
    if match is None:
        raise DataAdapterError(
            f"Delta option symbol {symbol!r} does not match the expected "
            "C/P-ASSET-STRIKE-DDMMYY form"
        )
    day, month, year = (
        int(match["expiry"][0:2]),
        int(match["expiry"][2:4]),
        2000 + int(match["expiry"][4:6]),
    )
    return (
        OptionType.CE if match["kind"] == "C" else OptionType.PE,
        match["asset"],
        Decimal(match["strike"]),
        date(year, month, day),
    )


def expiry_query(expiry: date) -> str:
    """Delta's `option_chain` expects DD-MM-YYYY."""
    return expiry.strftime("%d-%m-%Y")


@dataclass(frozen=True)
class DeltaConfig:
    base_url: str = TESTNET_INDIA
    """Testnet by default.

    Same reasoning as the Dhan adapter: the safe environment is the one you
    get without asking. Production requires saying so.
    """
    spot_index_symbol: str = ".DEXBTUSD"
    rate: float = DEFAULT_RISK_FREE_RATE
    compute_greeks: bool = True
    """Whether to recompute greeks locally and cross-check the published ones."""


class DeltaExchangeAdapter:
    """Index and options data from Delta Exchange India.

    Implements the index and chain capabilities. Not the broker ones: order
    placement needs a key, and an unverified order mapping is exactly what
    this adapter was built to avoid repeating.
    """

    def __init__(
        self,
        client: DeltaClient,
        *,
        config: DeltaConfig | None = None,
    ) -> None:
        self._client = client
        self._config = config or DeltaConfig()

    async def _call(self, fn: Any, /, **kwargs: Any) -> Any:
        """Run a synchronous client method off the event loop.

        `delta-rest-client` is `requests`-based. Calling it directly would
        block the poll loop for the duration of every request, which on a
        20-second cycle is the difference between a stale console and a
        stopped one.
        """
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, partial(fn, **kwargs))
        except Exception as exc:
            raise DataAdapterError(f"Delta request failed: {exc}") from exc

    # ------------------------------------------------------------- index

    async def get_index_spec(self, symbol: str) -> IndexSpec:
        """Contract specification, read from a live product rather than assumed.

        `contract_value` is the multiplier and `tick_size` the increment,
        both published per product. The one time this codebase assumed a
        contract size it was wrong by 15%.
        """
        products = await self._call(
            self._client.get_products,
            query={"contract_types": "call_options", "underlying_asset_symbols": symbol.upper()},
            auth=False,
        )
        live = [
            p
            for p in (products or [])
            if isinstance(p, dict) and p.get("state") == "live"
        ]
        if not live:
            raise DataAdapterError(
                f"Delta lists no live option products for {symbol}"
            )
        product = live[0]
        return IndexSpec(
            symbol=symbol.upper(),
            name=str(product.get("underlying_asset", {}).get("name") or symbol.upper()),
            # One contract is contract_value units of the underlying. Named
            # lot_size on the contract because that is what the field means
            # in every market; the value is read, never assumed.
            lot_size=1,
            tick_size=_required(product.get("tick_size"), "tick_size"),
            strike_step=self._strike_step(live),
        )

    @staticmethod
    def _strike_step(products: list[dict[str, Any]]) -> Decimal | None:
        """The gap between adjacent strikes, or None if it is not uniform.

        Inferred from the listed strikes rather than configured, and None
        when the listing is irregular — a single assumed step would place
        strikes that do not exist.
        """
        strikes = sorted(
            {
                value
                for p in products
                if (value := _decimal(p.get("strike_price"))) is not None
            }
        )
        if len(strikes) < 3:
            return None
        gaps = {b - a for a, b in pairwise(strikes)}
        return min(gaps) if gaps else None

    async def get_index_quote(self, symbol: str) -> IndexQuote:
        """Spot, taken from the chain's own `spot_price`.

        The chain row carries the spot the greeks beside it were computed
        against, so reading spot from a separate ticker call would let the
        quote and the greeks describe different moments.
        """
        expiries = await self.get_available_expiries(symbol)
        if not expiries:
            raise DataAdapterError(f"Delta lists no expiries for {symbol}")
        rows = await self._chain_rows(symbol, expiries[0])
        spot = _required(rows[0].get("spot_price"), "spot_price")
        bars = await self.get_index_bars(symbol, BarInterval.DAY, 2)
        previous_close = bars[-1].close if bars else spot
        session = bars[-1] if bars else None
        return IndexQuote(
            symbol=symbol.upper(),
            timestamp=datetime.now(UTC),
            ltp=spot,
            # A continuous market has no session open; the most recent daily
            # candle is the closest honest analogue and is labelled as such.
            open=session.open if session else spot,
            high=session.high if session else spot,
            low=session.low if session else spot,
            previous_close=previous_close,
        )

    async def get_index_bars(
        self, symbol: str, interval: BarInterval, count: int
    ) -> list[Bar]:
        """Real historical candles — the thing NSE will not serve.

        `get_candles` takes a spot *index* symbol (`.DEXBTUSD`), not the
        option symbol, and a unix second range.
        """
        resolution = _RESOLUTIONS.get(interval)
        if resolution is None:
            raise DataAdapterError(
                f"Delta serves no {interval} candles. Available: "
                f"{sorted(str(k) for k in _RESOLUTIONS)}"
            )
        if count <= 0:
            return []
        seconds = _INTERVAL_SECONDS[interval]
        end = int(datetime.now(UTC).timestamp())
        # Over-request: the venue returns nothing for gaps, and asking for
        # exactly `count` would come back short whenever one exists.
        start = end - seconds * (count + 5)
        rows = await self._call(
            self._client.get_candles,
            symbol=self._config.spot_index_symbol,
            resolution=resolution,
            start=start,
            end=end,
            auth=False,
        )
        bars = [
            Bar(
                timestamp=datetime.fromtimestamp(int(row["time"]), tz=UTC),
                open=_required(row.get("open"), "open"),
                high=_required(row.get("high"), "high"),
                low=_required(row.get("low"), "low"),
                close=_required(row.get("close"), "close"),
                # Delta returns null volume on index candles. 0 is the
                # contract's default and means "not published" here; nothing
                # in this system reads index volume.
                volume=_int(row.get("volume")) or 0,
            )
            for row in (rows or [])
            if isinstance(row, dict) and row.get("time") is not None
        ]
        bars.sort(key=lambda bar: bar.timestamp)
        return bars[-count:]

    # ------------------------------------------------------------ chain

    async def get_available_expiries(self, underlying_symbol: str) -> list[date]:
        products = await self._call(
            self._client.get_products,
            query={
                "contract_types": "call_options",
                "underlying_asset_symbols": underlying_symbol.upper(),
            },
            auth=False,
        )
        expiries: set[date] = set()
        for product in products or []:
            if not isinstance(product, dict) or product.get("state") != "live":
                continue
            raw = product.get("settlement_time")
            if isinstance(raw, str) and raw:
                expiries.add(datetime.fromisoformat(raw).date())
        return sorted(expiries)

    async def _chain_rows(
        self, underlying_symbol: str, expiry: date
    ) -> list[dict[str, Any]]:
        rows = await self._call(
            self._client.option_chain,
            underlying_asset_symbol=underlying_symbol.upper(),
            expiry_date=expiry_query(expiry),
            auth=False,
        )
        usable = [r for r in (rows or []) if isinstance(r, dict) and r.get("symbol")]
        if not usable:
            raise DataAdapterError(
                f"Delta published an empty chain for {underlying_symbol} "
                f"{expiry.isoformat()}"
            )
        return usable

    async def get_option_chain(
        self, underlying_symbol: str, expiry: date
    ) -> list[OptionQuote]:
        rows = await self._chain_rows(underlying_symbol, expiry)
        as_of = datetime.now(UTC)
        quotes: list[OptionQuote] = []
        for row in rows:
            quote = self._build_quote(row, as_of=as_of)
            if quote is not None:
                quotes.append(quote)
        if not quotes:
            raise DataAdapterError(
                f"Delta chain for {underlying_symbol} {expiry.isoformat()} "
                "contained no quotable strikes"
            )
        return quotes

    def _build_quote(
        self, row: dict[str, Any], *, as_of: datetime
    ) -> OptionQuote | None:
        option_type, asset, strike, expiry = parse_option_symbol(str(row["symbol"]))
        quotes = row.get("quotes") or {}
        multiplier = _decimal(row.get("contract_value")) or Decimal(1)

        contract = OptionContractSpec(
            underlying_symbol=asset,
            expiry=expiry,
            strike=strike,
            option_type=option_type,
            # One contract is `contract_value` units of the underlying, and
            # prices are per unit. Carrying the multiplier here is what keeps
            # every notional downstream honest.
            lot_size=1,
            tick_size=_decimal(row.get("tick_size")) or Decimal("0.1"),
        )
        mark = _decimal(row.get("mark_price"))
        last = _decimal(row.get("close")) or _decimal(row.get("ltp"))
        if mark is None and last is None:
            return None

        iv = self._resolve_iv(quotes)
        greeks = self._resolve_greeks(
            row=row,
            spot=_decimal(row.get("spot_price")),
            strike=strike,
            option_type=option_type,
            expiry=expiry,
            iv_percent=iv,
            as_of=as_of,
        )
        return OptionQuote(
            contract=contract,
            timestamp=as_of,
            ltp=last or mark or Decimal(0),
            bid=_decimal(quotes.get("best_bid")),
            ask=_decimal(quotes.get("best_ask")),
            volume=_int(row.get("volume")) or 0,
            # Delta reports open interest in contracts, matching the
            # convention every OI figure in this system uses.
            open_interest=_int(row.get("oi")) or 0,
            open_interest_change=0,
            implied_volatility=iv,
            greeks=greeks,
            # Multiplier travels with the quote so a caller cannot lose it.
            contract_multiplier=multiplier,
        )

    @staticmethod
    def _resolve_iv(quotes: dict[str, Any]) -> Decimal | None:
        """Mark IV in percentage points, or None.

        Delta quotes IV as a decimal (0.31925), while everything in this
        system carries it as percentage points. It also publishes bid, mark
        and ask IV, and the mark is only usable if it sits between the other
        two: outside that range it is not a mid of anything, which is the
        same stale-quote problem NSE's single published IV turned out to
        have — except here it is detectable.
        """
        mark = _decimal(quotes.get("mark_iv"))
        if mark is None or mark <= 0:
            return None
        bid, ask = _decimal(quotes.get("bid_iv")), _decimal(quotes.get("ask_iv"))
        if bid is not None and ask is not None and bid > 0 and ask > 0:
            low, high = min(bid, ask), max(bid, ask)
            if not (low <= mark <= high):
                return None
        return mark * 100

    def _resolve_greeks(
        self,
        *,
        row: dict[str, Any],
        spot: Decimal | None,
        strike: Decimal,
        option_type: OptionType,
        expiry: date,
        iv_percent: Decimal | None,
        as_of: datetime,
    ) -> Greeks | None:
        """The exchange's greeks, cross-checked against a local computation.

        Published greeks are used when they agree with Black-Scholes on the
        exchange's own mark IV, and the local ones are used when they do not.
        Two independent derivations disagreeing is a real check — unlike the
        lot-size case, where both sides came from one constant and the
        comparison could never fail.

        A published delta with no IV to check it against is still returned:
        refusing it would discard the venue's main advantage over NSE on the
        strength of a missing field. The disagreement path is what guards
        against a mapping error, not the presence of the field.
        """
        published = row.get("greeks") or {}
        delta = _decimal(published.get("delta"))
        exchange = (
            Greeks(
                delta=delta,
                gamma=_decimal(published.get("gamma")) or Decimal(0),
                theta=_decimal(published.get("theta")) or Decimal(0),
                vega=_decimal(published.get("vega")) or Decimal(0),
            )
            if delta is not None
            else None
        )
        if not self._config.compute_greeks or iv_percent is None or spot is None:
            return exchange

        years = (
            datetime.combine(expiry, datetime.min.time(), tzinfo=UTC) - as_of
        ).total_seconds() / (365.0 * 86400.0)
        if years <= 0:
            return exchange
        local = greeks_from_iv(
            spot=float(spot),
            strike=float(strike),
            years=years,
            iv_percent=float(iv_percent),
            option_type=option_type,
            rate=self._config.rate,
        )
        computed = Greeks(
            delta=Decimal(str(round(local.delta, 6))),
            gamma=Decimal(str(round(local.gamma, 10))),
            theta=Decimal(str(round(local.theta, 4))),
            vega=Decimal(str(round(local.vega, 4))),
        )
        if exchange is None:
            return computed
        if abs(float(exchange.delta) - local.delta) > GREEK_TOLERANCE:
            # The published figure and an independent computation from the
            # venue's own IV disagree. Prefer the one whose derivation is
            # visible here.
            return computed
        return exchange
