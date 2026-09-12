"""Live adapter for NSE's daily F&O bhavcopy — a real historical option chain.

Why this exists
---------------
`backtest/replay.py` and `nse_archive.py` both state that NSE publishes no
historical option chains and that no free source does. **That is wrong**, and
the error was load-bearing: it capped the backtest at "signal quality only",
so in 246 replayed sessions the Strategy Engine returned NO_TRADE 216 times
out of 216 — not because the strategy declined, but because a strike cannot be
chosen from a chain that was never loaded. The economics of the whole system
were therefore unmeasured, and read as a decision.

NSE publishes a UDiFF bhavcopy for every trading session at

    https://nsearchives.nseindia.com/content/fo/
        BhavCopy_NSE_FO_0_0_0_<YYYYMMDD>_F_0000.csv.zip

~1 MB zipped, ~33,700 rows, free, official, no login. Per contract it carries
strike, expiry, option type, OHLC, settlement price, open interest, change in
open interest, traded volume, number of trades, the underlying's price, and
the board lot. That is a daily option chain.

What it cannot tell you, and why that matters more than what it can
-------------------------------------------------------------------
* **No bid or ask.** A bhavcopy is an end-of-day summary, not a book. Quotes
  are built with `bid=None, ask=None`, so `OptionQuote.mid` falls back to the
  settlement price and `relative_spread` returns `None` — *unmeasured*, never
  zero. This is the single most important property of this module. A spread
  of zero would tell the Strike Engine's liquidity filter that every strike is
  perfectly tight, and a backtest that believes it can trade a 21700-strike
  put at its settlement price will invent an edge that does not exist.
* **A settlement price is not a fill.** Nobody transacts at it. A backtest
  priced from settlement is *optimistic by construction*, which makes its
  failures conclusive and its successes merely suggestive. Read a loss here as
  final and a profit here as "now go measure slippage".
* **No implied volatility.** NSE does not publish it. Left `None` rather than
  solved silently; `analytics/pricing.py` can invert the settlement price when
  a caller explicitly wants that, and the assumption then belongs to them.
* **One chain per session, at the close.** Same resolution as the daily index
  archive, so a replayed cycle stays one decision per day.
* **UDiFF only.** NSE moved to this format in July 2024; the older
  `fo<DDMMMYYYY>bhav.csv.zip` files are a different layout on a different
  host. `EARLIEST_UDIFF_SESSION` is the floor and an earlier date is refused
  rather than returned empty, because "no rows" and "wrong decade" must not
  look alike.

Caching
-------
A year of sessions is ~250 requests of ~1 MB. Re-fetching those on every
replay would be both slow and abusive, so a fetched day is written under
`cache_dir` and served from there afterwards. Only successful, non-empty
payloads are cached: caching a 403 interstitial would make a transient block
permanent.
"""

from __future__ import annotations

import asyncio
import csv
import io
import zipfile
from collections.abc import Iterable, Sequence
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation
from pathlib import Path

from index_option_brain.contracts.enums import OptionType
from index_option_brain.contracts.instruments import OptionContractSpec, OptionQuote
from index_option_brain.contracts.market_state import OptionsState
from index_option_brain.data.adapters.base import DataAdapterError
from index_option_brain.data.http import HttpError, HttpSession, HttpxSession

ARCHIVE_BASE = "https://nsearchives.nseindia.com/content/fo"

#: NSE's UDiFF bhavcopy starts here. Before this the F&O archive is a
#: different filename, host and column set.
EARLIEST_UDIFF_SESSION = date(2024, 7, 8)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "*/*",
    # The archive host 403s a request with no referer often enough that
    # omitting it turns a working day into an apparent holiday.
    "Referer": "https://www.nseindia.com/all-reports-derivatives",
}

_OPTION_TYPES = {"CE": OptionType.CE, "PE": OptionType.PE}


def bhavcopy_url(day: date) -> str:
    return f"{ARCHIVE_BASE}/BhavCopy_NSE_FO_0_0_0_{day.strftime('%Y%m%d')}_F_0000.csv.zip"


def _decimal(raw: str | None) -> Decimal | None:
    """A Decimal, or None for a blank or unparseable cell.

    None rather than zero: a missing settlement price and a settlement price
    of zero are different facts, and only one of them is tradeable.
    """
    if raw is None:
        return None
    text = raw.strip()
    if not text or text in {"-", "NA"}:
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def _int(raw: str | None) -> int:
    value = _decimal(raw)
    return int(value) if value is not None else 0


def unzip_bhavcopy(payload: bytes, *, day: date) -> str:
    """The CSV inside NSE's zip.

    Raises rather than returning empty when the payload is not a zip, because
    the failure mode being guarded against is NSE answering with an HTML
    interstitial and HTTP 200 — which would otherwise parse to a chain of
    zero contracts and read as a quiet session.
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as exc:
        head = payload[:120].decode("utf-8", "replace")
        raise DataAdapterError(
            f"NSE F&O bhavcopy for {day.isoformat()} is not a zip "
            f"({len(payload)} bytes, starts {head!r}). This is usually an "
            f"anti-bot interstitial served with HTTP 200."
        ) from exc
    names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
    if not names:
        raise DataAdapterError(
            f"NSE F&O bhavcopy for {day.isoformat()} contains no CSV: {archive.namelist()}"
        )
    return archive.read(names[0]).decode("utf-8", "replace")


def parse_bhavcopy(
    body: str,
    *,
    underlying: str,
    day: date,
    expiry: date | None = None,
) -> OptionsState:
    """An `OptionsState` for `underlying` from one session's bhavcopy.

    `expiry` defaults to the nearest expiry on or after `day`. Every expiry
    present is reported in `available_expiries`, so a caller that wants the
    monthly rather than the weekly can pick without a second fetch.
    """
    symbol = underlying.strip().upper()
    rows = [
        row
        for row in csv.DictReader(io.StringIO(body))
        if (row.get("TckrSymb") or "").strip().upper() == symbol
        and (row.get("OptnTp") or "").strip().upper() in _OPTION_TYPES
    ]
    if not rows:
        raise DataAdapterError(
            f"NSE F&O bhavcopy for {day.isoformat()} carries no {symbol} options. "
            f"Check the symbol: the file spells the underlying as it trades "
            f"(NIFTY, BANKNIFTY, FINNIFTY), not as the index is named."
        )

    expiries = sorted({d for row in rows if (d := _date(row.get("XpryDt"))) is not None})
    if expiry is None:
        forward_expiries = [d for d in expiries if d >= day]
        if not forward_expiries:
            raise DataAdapterError(
                f"Every {symbol} expiry in the {day.isoformat()} bhavcopy has "
                f"already passed (latest {expiries[-1] if expiries else 'none'}). "
                f"A chain cannot be built from expired contracts."
            )
        expiry = forward_expiries[0]
    elif expiry not in expiries:
        raise DataAdapterError(
            f"No {symbol} contracts expiring {expiry.isoformat()} in the "
            f"{day.isoformat()} bhavcopy. Available: {[d.isoformat() for d in expiries]}"
        )

    # The close is what a daily bar describes, so the chain is timestamped at
    # it rather than at midnight — the same convention the index archive uses.
    stamp = datetime.combine(day, time(15, 30))
    chain: list[OptionQuote] = []
    for row in rows:
        if _date(row.get("XpryDt")) != expiry:
            continue
        strike = _decimal(row.get("StrkPric"))
        settle = _decimal(row.get("SttlmPric")) or _decimal(row.get("ClsPric"))
        if strike is None or settle is None:
            # A contract with no strike or no settlement is not a quote. Skipped
            # rather than defaulted, so an unpriced strike cannot be selected.
            continue
        chain.append(
            OptionQuote(
                contract=OptionContractSpec(
                    underlying_symbol=symbol,
                    expiry=expiry,
                    strike=strike,
                    option_type=_OPTION_TYPES[(row["OptnTp"]).strip().upper()],
                    lot_size=_int(row.get("NewBrdLotQty")) or 1,
                    tick_size=Decimal("0.05"),
                ),
                timestamp=stamp,
                ltp=settle,
                # Not a book. See the module docstring: None keeps
                # `relative_spread` unmeasured instead of perfectly tight.
                bid=None,
                ask=None,
                volume=_int(row.get("TtlTradgVol")),
                open_interest=_int(row.get("OpnIntrst")),
                open_interest_change=_int(row.get("ChngInOpnIntrst")),
                implied_volatility=None,
            )
        )

    if not chain:
        raise DataAdapterError(
            f"Every {symbol} contract expiring {expiry.isoformat()} in the "
            f"{day.isoformat()} bhavcopy was unpriced."
        )
    return OptionsState(
        chain=sorted(chain, key=lambda q: (q.contract.strike, q.contract.option_type)),
        expiry=expiry,
        available_expiries=expiries,
    )


def _date(raw: str | None) -> date | None:
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


class NseFoArchiveAdapter:
    """Daily option chains from NSE's published F&O bhavcopy."""

    def __init__(
        self,
        *,
        session: HttpSession | None = None,
        cache_dir: Path | str | None = None,
        max_attempts: int = 4,
        retry_pause: float = 1.5,
        polite_pause: float = 0.4,
    ) -> None:
        self._session: HttpSession = session or HttpxSession(headers=_HEADERS, timeout=60.0)
        self._cache = Path(cache_dir) if cache_dir is not None else None
        self._max_attempts = max_attempts
        self._retry_pause = retry_pause
        # A bulk fetch that hammers the archive is what triggers the throttle
        # in the first place, so a cache miss pauses before the next request.
        self._polite_pause = polite_pause
        if self._cache is not None:
            self._cache.mkdir(parents=True, exist_ok=True)

    async def aclose(self) -> None:
        await self._session.aclose()

    def _cached(self, day: date) -> bytes | None:
        if self._cache is None:
            return None
        path = self._cache / f"{day.isoformat()}.zip"
        return path.read_bytes() if path.is_file() else None

    def _store(self, day: date, payload: bytes) -> None:
        if self._cache is None or not payload:
            return
        (self._cache / f"{day.isoformat()}.zip").write_bytes(payload)

    async def fetch_raw(self, day: date) -> bytes | None:
        """One session's zipped bhavcopy, or `None` when NSE published none.

        The hard part is that the archive answers **403 for both** "no such
        file" and "you are asking too fast", with an identical 482-byte body.
        Mapping 403 to "no session" — which the first version of this adapter
        did — means a rate-limited weekday silently vanishes from the backtest
        and the result still looks complete. That was measured, not feared: a
        run asking for five sessions got 403 on two weekdays, and every one of
        them returned 200 with ~1 MB when asked again a moment later.

        So a 403 is retried with a growing pause, and a weekday that still
        refuses after `max_attempts` raises. Only a weekend — which is never a
        trading day on any calendar — returns `None` without asking, and a
        weekday 404 returns `None` as a genuine exchange holiday.
        """
        if day < EARLIEST_UDIFF_SESSION:
            raise DataAdapterError(
                f"{day.isoformat()} predates NSE's UDiFF bhavcopy "
                f"({EARLIEST_UDIFF_SESSION.isoformat()}). Earlier sessions exist "
                f"in the retired fo<DDMMMYYYY>bhav.csv format, which this "
                f"adapter does not read."
            )
        if day.weekday() >= 5:
            return None
        cached = self._cached(day)
        if cached is not None:
            return cached or None

        last_status = None
        for attempt in range(self._max_attempts):
            if attempt:
                await asyncio.sleep(self._retry_pause * attempt)
            try:
                response = await self._session.get(bhavcopy_url(day))
            except HttpError as exc:
                raise DataAdapterError(
                    f"Could not reach NSE for the {day.isoformat()} bhavcopy: {exc}"
                ) from exc
            last_status = response.status_code
            if response.status_code == 404:
                return None
            if response.status_code == 403:
                continue
            if not response.is_ok:
                raise DataAdapterError(
                    f"NSE answered {response.status_code} for the "
                    f"{day.isoformat()} bhavcopy"
                )
            payload = response.content
            if not payload:
                raise DataAdapterError(
                    f"NSE answered 200 with an empty body for the {day.isoformat()} "
                    f"bhavcopy. An empty payload is not an empty session."
                )
            self._store(day, payload)
            return payload

        raise DataAdapterError(
            f"NSE answered {last_status} for the {day.isoformat()} bhavcopy on all "
            f"{self._max_attempts} attempts. On a weekday this is ambiguous — the "
            f"archive returns 403 both for a file that does not exist and for a "
            f"client it is throttling — so it is raised rather than recorded as a "
            f"holiday. Re-run; a throttle clears in seconds."
        )

    async def get_option_chain(
        self,
        underlying: str,
        day: date,
        *,
        expiry: date | None = None,
    ) -> OptionsState | None:
        """The chain at the close of `day`, or `None` if NSE published no session."""
        payload = await self.fetch_raw(day)
        if payload is None:
            return None
        body = unzip_bhavcopy(payload, day=day)
        return parse_bhavcopy(body, underlying=underlying, day=day, expiry=expiry)

    async def get_price_map(
        self, underlying: str, day: date
    ) -> dict[tuple[date, Decimal, OptionType], Decimal] | None:
        """Settlement price of every `underlying` option that traded on `day`.

        Keyed across **all** expiries, not just the nearest, because this is
        what prices an exit: a structure opened against the 15 Sep expiry is
        closed on some later session when that expiry is no longer the front
        one. `get_option_chain` deliberately returns a single expiry, and using
        it to price exits would silently drop every trade held past a roll.

        `None` for a session NSE did not publish.
        """
        payload = await self.fetch_raw(day)
        if payload is None:
            return None
        symbol = underlying.strip().upper()
        prices: dict[tuple[date, Decimal, OptionType], Decimal] = {}
        body = unzip_bhavcopy(payload, day=day)
        for row in csv.DictReader(io.StringIO(body)):
            if (row.get("TckrSymb") or "").strip().upper() != symbol:
                continue
            opt = (row.get("OptnTp") or "").strip().upper()
            if opt not in _OPTION_TYPES:
                continue
            expiry = _date(row.get("XpryDt"))
            strike = _decimal(row.get("StrkPric"))
            settle = _decimal(row.get("SttlmPric")) or _decimal(row.get("ClsPric"))
            if expiry is None or strike is None or settle is None:
                continue
            prices[(expiry, strike, _OPTION_TYPES[opt])] = settle
        return prices

    async def get_many_option_chains(
        self,
        underlying: str,
        days: Sequence[date] | Iterable[date],
    ) -> tuple[dict[date, OptionsState], list[date]]:
        """Chains for every session NSE published among `days`.

        Returns the chains **and the weekdays it could not resolve**, rather
        than a dict whose length you would have to trust. A backtest silently
        missing 40 sessions and one missing 12 exchange holidays look identical
        from the outside, and only one of them is fine.
        """
        chains: dict[date, OptionsState] = {}
        unresolved: list[date] = []
        for day in days:
            miss = self._cached(day) is None and day.weekday() < 5
            try:
                state = await self.get_option_chain(underlying, day)
            except DataAdapterError:
                unresolved.append(day)
                continue
            if state is not None:
                chains[day] = state
            if miss and self._polite_pause:
                await asyncio.sleep(self._polite_pause)
        return chains, unresolved
