"""Live adapter for NSE's daily index archive.

This is a **live** adapter over an official NSE publication: the end-of-day
index file the exchange writes to `archives.nseindia.com` for every trading
session. Nothing here is simulated. A session NSE did not publish yields no
bar rather than an interpolated one.

Why this exists
---------------
`NsePublicAdapter` documents that `/api/historical/indicesHistory` answers an
automated client with an anti-bot interstitial, so the Index brain had no
daily history and its confidence stayed pinned at 0.00 — which the Regime
Engine's coverage gate correctly refused to classify from. The archive is a
different surface: a plain CSV on a static host, no cookie warm-up, no
interstitial, HTTP 200 for any published session.

What it serves
--------------
One file per trading day, containing every NSE index with its open, high,
low, close, volume, turnover, P/E, P/B and dividend yield. The file for
`ind_close_all_02092026.csv` carries NIFTY 50 at 23858 / 23914.45 / 23786.8 /
23914.45 — the session's true daily bar, including the close the brains use
as the previous-day reference.

What it does not serve
----------------------
* **No intraday.** One bar per session, nothing finer. Intraday bars still
  come from aggregating live snapshots forward, or from a broker.
* **No current session.** The file for a day appears after that session
  closes. `get_index_bars` therefore never returns a partial candle, which
  is exactly the guarantee `IndexDataAdapter.get_index_bars` requires.
* **No quotes, no chain, no account.** It is a history source only, and
  implements only the bar capability of the index interface.

Operational notes
-----------------
* Weekends and exchange holidays have no file. The fetch walks calendar days
  backwards and treats a 404 as "not a trading day", never as an error.
* Days are fetched concurrently and bounded by a semaphore: a cold start
  needs ~60 files, and doing that serially costs a minute of startup.
* Index names are matched case-insensitively against the file's `Index Name`
  column ("Nifty 50", not "NIFTY 50"), because the archive's capitalisation
  differs from the live API's.
"""

from __future__ import annotations

import asyncio
import csv
import io
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

from index_option_brain.contracts.enums import BarInterval
from index_option_brain.contracts.instruments import Bar
from index_option_brain.data.adapters.base import DataAdapterError
from index_option_brain.data.http import HttpSession, HttpxSession

ARCHIVE_BASE = "https://archives.nseindia.com/content/indices"

#: The live API and the archive spell the same index differently.
ARCHIVE_INDEX_NAMES: dict[str, str] = {
    "NIFTY": "nifty 50",
    "NIFTY 50": "nifty 50",
    "BANKNIFTY": "nifty bank",
    "NIFTY BANK": "nifty bank",
    "FINNIFTY": "nifty financial services",
    "MIDCPNIFTY": "nifty midcap select",
    # The archive carries India VIX as an index row like any other, which is
    # what makes the Volatility brain replayable on real history.
    "INDIAVIX": "india vix",
    "INDIA VIX": "india vix",
}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/csv,text/plain,*/*",
}


def archive_index_name(symbol: str) -> str:
    """The archive's spelling of `symbol`, lower-cased for comparison."""
    key = symbol.strip().upper()
    resolved = ARCHIVE_INDEX_NAMES.get(key)
    if resolved is None:
        raise DataAdapterError(
            f"No NSE archive index name is known for {symbol!r}. "
            f"Known: {sorted(ARCHIVE_INDEX_NAMES)}"
        )
    return resolved


def archive_url(day: date) -> str:
    return f"{ARCHIVE_BASE}/ind_close_all_{day.strftime('%d%m%Y')}.csv"


def _decimal(row: dict[str, str], column: str, day: date) -> Decimal:
    raw = (row.get(column) or "").strip().replace(",", "")
    if not raw or raw == "-":
        raise DataAdapterError(f"NSE archive {day.isoformat()} has no {column}")
    try:
        return Decimal(raw)
    except InvalidOperation as exc:
        raise DataAdapterError(
            f"NSE archive {day.isoformat()} has an unparseable {column}: {raw!r}"
        ) from exc


def _volume(row: dict[str, str]) -> int:
    raw = (row.get("Volume") or "").strip().replace(",", "")
    if not raw or raw == "-":
        return 0
    try:
        return int(Decimal(raw))
    except InvalidOperation:
        return 0


def parse_archive_csv(body: str, *, index_name: str, day: date) -> Bar | None:
    """The one index's bar out of a whole-market archive file, or None.

    None means the file was served but did not carry this index — a real
    outcome for an index introduced after the file's date, and one that must
    not be confused with a missing file.
    """
    for row in csv.DictReader(io.StringIO(body)):
        name = (row.get("Index Name") or "").strip().lower()
        if name != index_name:
            continue
        return Bar(
            # The archive's own date column is authoritative; the requested
            # day is only how the file was addressed.
            timestamp=datetime.combine(day, datetime.min.time(), tzinfo=UTC),
            open=_decimal(row, "Open Index Value", day),
            high=_decimal(row, "High Index Value", day),
            low=_decimal(row, "Low Index Value", day),
            close=_decimal(row, "Closing Index Value", day),
            volume=_volume(row),
        )
    return None


def parse_archive_csv_many(
    body: str, *, index_names: set[str], day: date
) -> dict[str, Bar]:
    """Every requested index's bar out of one whole-market archive file.

    Indices the file does not carry are simply absent from the mapping — the
    same distinction `parse_archive_csv` makes with None, kept per index so a
    file that carries NIFTY but not India VIX yields the one it has.
    """
    found: dict[str, Bar] = {}
    for row in csv.DictReader(io.StringIO(body)):
        name = (row.get("Index Name") or "").strip().lower()
        if name not in index_names or name in found:
            continue
        found[name] = Bar(
            timestamp=datetime.combine(day, datetime.min.time(), tzinfo=UTC),
            open=_decimal(row, "Open Index Value", day),
            high=_decimal(row, "High Index Value", day),
            low=_decimal(row, "Low Index Value", day),
            close=_decimal(row, "Closing Index Value", day),
            volume=_volume(row),
        )
    return found


class _Unresolved:
    """A day the archive could not be read — neither a bar nor a known holiday."""

    __slots__ = ()


_UNRESOLVED = _Unresolved()


class NseArchiveAdapter:
    """Daily index bars from NSE's published end-of-day archive.

    Implements the bar capability alone. It is deliberately not an
    `IndexDataAdapter`: it serves no quote and no spec, and claiming that
    interface would advertise capabilities it does not have.
    """

    def __init__(
        self,
        session: HttpSession | None = None,
        *,
        concurrency: int = 4,
        max_lookback_days: int = 400,
        attempts: int = 3,
        backoff: float = 0.5,
        max_unresolved: int = 2,
    ) -> None:
        """
        `concurrency` is deliberately low: the archive rate-limits, and a
        throttled request is indistinguishable from a dead one at this layer.
        `max_unresolved` is the number of unreadable days tolerated before the
        series is refused outright — a couple of holes at the far end of a
        60-bar window barely move an EMA, but a series full of them is not
        history and must not be presented as such.
        """
        self._session: HttpSession = session or HttpxSession(headers=_HEADERS)
        self._semaphore = asyncio.Semaphore(concurrency)
        self._max_lookback_days = max_lookback_days
        self._attempts = attempts
        self._backoff = backoff
        self._max_unresolved = max_unresolved

    async def aclose(self) -> None:
        await self._session.aclose()

    async def get_many_index_bars(
        self,
        symbols: Sequence[str],
        *,
        count: int = 60,
        as_of: date | None = None,
    ) -> dict[str, list[Bar]]:
        """Daily bars for several indices, fetching each archive file once.

        One file holds every NSE index, so asking for NIFTY and India VIX
        separately would double the request count against a host that rate
        limits — and the two series could then disagree about which sessions
        existed, which is worse than slow. Returned series are aligned by
        construction: they come from the same set of files.

        A symbol the archive never carried is absent from the mapping rather
        than present and empty.
        """
        if not symbols or count <= 0:
            return {}
        names = {symbol: archive_index_name(symbol) for symbol in symbols}
        end = as_of or datetime.now(UTC).date()
        span = min(self._max_lookback_days, int(count * 7 / 5) + 14)
        candidates = [
            end - timedelta(days=offset)
            for offset in range(span)
            if (end - timedelta(days=offset)).weekday() < 5
        ]

        results = await asyncio.gather(
            *(self._fetch_day(set(names.values()), day) for day in candidates)
        )
        paired = sorted(zip(candidates, results, strict=True), key=lambda p: p[0])

        unresolved = [d for d, r in paired if isinstance(r, _Unresolved)]
        out: dict[str, list[Bar]] = {}
        for symbol, name in names.items():
            series = [
                r[name]
                for _, r in paired
                if isinstance(r, dict) and name in r
            ]
            if series:
                out[symbol] = series[-count:]

        if not out:
            raise DataAdapterError(
                f"NSE archive returned no usable session for {list(symbols)} in "
                f"the {span} days to {end.isoformat()}"
            )
        shortest = min(len(v) for v in out.values())
        inside = [
            day
            for day in unresolved
            if shortest < count
            or day >= min(v[0].timestamp.date() for v in out.values())
        ]
        if len(inside) > self._max_unresolved:
            raise DataAdapterError(
                f"NSE archive left {len(inside)} day(s) unreadable in the "
                f"{count} sessions to {end.isoformat()}; shortest series "
                f"resolved {shortest} bars. Refusing to return series with "
                f"holes — an indicator cannot tell one from a complete series."
            )
        return out

    async def _fetch_day(
        self, symbol_names: str | set[str], day: date
    ) -> dict[str, Bar] | Bar | None | _Unresolved:
        """One session's bar, `None` for a non-trading day, `_UNRESOLVED` otherwise.

        The three outcomes must stay distinct. A 404 is the archive's way of
        saying "no session that day" — weekends and exchange holidays both —
        and is a fact. A timeout or a 5xx says nothing about whether a session
        happened, and collapsing it into `None` would turn a transport failure
        into a silent hole: the caller would receive a shorter series that
        looks exactly like a complete one, and every indicator computed over
        it would be confidently wrong. That bug was observed here — two
        consecutive cold starts produced RSI(14) of 27.6 and 35.1 on the same
        market, because different days happened to fail each time.
        """
        for attempt in range(self._attempts):
            async with self._semaphore:
                try:
                    response: Any = await self._session.get(archive_url(day))
                    status = getattr(response, "status_code", 200)
                except Exception:  # noqa: BLE001 - see below
                    # Deliberately blind: every transport failure means the
                    # same thing here — this day is unresolved — and the set
                    # is open-ended (timeouts, DNS, TLS, proxy, connection
                    # reset). Narrowing it would let an unlisted error escape
                    # and fail a 60-request cold start over one bad day,
                    # which is the outcome the retry exists to prevent. The
                    # failure is not swallowed: it becomes _UNRESOLVED, and
                    # get_index_bars refuses the series if too many do.
                    status = None
                    response = None
            if status == 404:
                return None
            if status == 200:
                body = getattr(response, "text", "")
                if not body or "Index Name" not in body:
                    return _UNRESOLVED
                try:
                    if isinstance(symbol_names, str):
                        return parse_archive_csv(
                            body, index_name=symbol_names, day=day
                        )
                    return parse_archive_csv_many(
                        body, index_names=symbol_names, day=day
                    )
                except DataAdapterError:
                    return _UNRESOLVED
            if attempt + 1 < self._attempts:
                await asyncio.sleep(self._backoff * (2**attempt))
        return _UNRESOLVED

    async def get_index_bars(
        self,
        symbol: str,
        interval: BarInterval = BarInterval.DAY,
        count: int = 60,
        *,
        as_of: date | None = None,
    ) -> list[Bar]:
        """The most recent `count` completed daily bars, oldest first.

        `as_of` defaults to today; the current session has no archive file, so
        the newest bar returned is always the previous close — the reference
        the brains expect and never a forming candle.
        """
        if interval is not BarInterval.DAY:
            raise DataAdapterError(
                f"The NSE archive publishes daily bars only, not {interval}. "
                "Intraday bars come from live aggregation or a broker."
            )
        if count <= 0:
            return []

        index_name = archive_index_name(symbol)
        end = as_of or datetime.now(UTC).date()

        # Walk back over calendar days, asking for more than `count` because
        # weekends and holidays yield nothing. The 7/5 ratio covers weekends;
        # the +14 covers a holiday-dense stretch such as Diwali week.
        span = min(self._max_lookback_days, int(count * 7 / 5) + 14)
        days = [end - timedelta(days=offset) for offset in range(span)]
        candidates = [day for day in days if day.weekday() < 5]

        results = await asyncio.gather(*(self._fetch_day(index_name, day) for day in candidates))

        # Pair each day with its outcome so an unreadable day can be located,
        # not merely counted: a hole next to the live session corrupts the
        # previous-close reference, while one 80 sessions back does not.
        paired = sorted(zip(candidates, results, strict=True), key=lambda p: p[0])
        bars = [r for _, r in paired if isinstance(r, Bar)]
        unresolved = [d for d, r in paired if isinstance(r, _Unresolved)]

        if not bars:
            raise DataAdapterError(
                f"NSE archive returned no usable session for {symbol} in the "
                f"{span} days to {end.isoformat()}"
            )

        # Which unreadable days actually corrupt the answer? Defining the
        # window as "the span of the bars we got" is circular — a day that
        # failed is by construction not among the successes, so every hole
        # would fall outside it and the check would never fire. Two cases
        # count instead: a hole *between* bars we kept (the series has a gap),
        # and any hole at all when the series came back short (that day is
        # exactly what would have filled it).
        kept = bars[-count:]
        window_start = kept[0].timestamp.date()
        inside = [day for day in unresolved if day >= window_start or len(kept) < count]
        if len(inside) > self._max_unresolved:
            raise DataAdapterError(
                f"NSE archive left {len(inside)} day(s) unreadable in the "
                f"{count} sessions to {end.isoformat()} "
                f"({inside[0].isoformat()} … {inside[-1].isoformat()}); "
                f"{len(kept)} bars resolved. Refusing to return a series with "
                f"holes — an indicator cannot tell one from a complete series."
            )
        return bars[-count:]
