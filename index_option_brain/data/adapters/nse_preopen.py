"""Live adapter for NSE's pre-open session, as a constituent breadth feed.

This is a **live** adapter over `/api/market-data-pre-open`, the endpoint
behind the exchange's own pre-open board. Nothing here is simulated.

Why this exists
---------------
`NsePublicAdapter` documents that `/api/equity-stockIndices` returns 404 for
this client, so index breadth had no source and the Constituent brain
reported "No constituent quotes available" with confidence 0.00 on every
cycle. The pre-open endpoint answers, and carries all 50 NIFTY constituents
with their indicative equilibrium price, previous close, matched quantity and
market capitalisation.

Weights are derived from `marketCap`. NSE weights its indices by free-float
capitalisation, not total, so this would be the wrong field if the payload
carried the total — it does not appear to. Reconstructing the index from
these weights and each constituent's IEP on 3 Sep 2026 gave +0.350% against
the exchange's own published +0.35% (23,998.11 vs 23,997.95, 0.16 points
apart), which is the accuracy of a free-float series and not of a total-cap
approximation.

What it serves
--------------
* All constituents of NIFTY, BANKNIFTY, or the whole F&O universe, keyed by
  `key=NIFTY` / `BANKNIFTY` / `FO` / `ALL`.
* Per constituent: IEP, previous close, matched quantity, turnover, market
  cap, 52-week range.
* The exchange's own advance/decline/unchanged counts and the indicative
  index open, which are exposed via `get_pre_open_summary`.

What it does not serve
----------------------
* **No sector.** The payload carries none, so `ConstituentSpec.sector` is
  reported as UNKNOWN rather than guessed from the symbol. Sector rotation
  stays unavailable until a source for it exists; a wrong sector map would
  produce confident rotation analysis of a partition that does not exist.
* **No intraday updates.** The pre-open auction runs 09:00-09:08 and matches
  by 09:12. After that the snapshot is frozen: it describes the open, not the
  current market.
* **No open/high/low.** An auction clears at one price, so `open`, `high`,
  `low` and `ltp` are all the IEP. That is the literal truth of the session,
  not a fill-in.

Staleness
---------
The frozen snapshot is the hazard this adapter exists to manage. Breadth at
the open genuinely informs the early session, but a 09:07 auction is not a
measurement of the market at 14:00, and serving it as one would put a
confident breadth reading behind every afternoon decision. Quotes are
therefore refused once the payload's own timestamp is older than
`max_staleness`, and the Constituent brain degrades to its honest
"no quotes" state rather than reading a morning snapshot as current.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

from index_option_brain.contracts.instruments import ConstituentQuote, ConstituentSpec
from index_option_brain.data.adapters.base import (
    ConstituentDataAdapter,
    DataAdapterError,
)
from index_option_brain.data.adapters.nse_public import NSE_BASE, parse_ist_timestamp

PRE_OPEN_URL = f"{NSE_BASE}/api/market-data-pre-open"

#: The endpoint's `key` for each index this system trades.
PRE_OPEN_KEYS: dict[str, str] = {
    "NIFTY": "NIFTY",
    "NIFTY 50": "NIFTY",
    "BANKNIFTY": "BANKNIFTY",
    "NIFTY BANK": "BANKNIFTY",
}

#: Reported rather than guessed. See the module docstring.
UNKNOWN_SECTOR = "UNKNOWN"

#: The auction matches by 09:12 and the snapshot never changes after that, so
#: the default keeps it through the opening range and drops it afterwards.
DEFAULT_MAX_STALENESS = timedelta(minutes=45)


@dataclass(frozen=True)
class PreOpenSummary:
    """The exchange's own view of the auction, not a reconstruction of it.

    `indicative_open` is NSE's published figure. It is kept alongside the
    constituent rows precisely so a reconstruction from those rows can be
    checked against it rather than trusted.
    """

    timestamp: datetime
    advances: int
    declines: int
    unchanged: int
    indicative_open: Decimal | None
    indicative_change: Decimal | None
    indicative_change_pct: float | None
    total_traded_value: Decimal | None

    @property
    def participation(self) -> int:
        return self.advances + self.declines + self.unchanged

    @property
    def advance_decline_ratio(self) -> float | None:
        """Advances per decline, or None when nothing declined.

        None rather than infinity: a ratio with a zero denominator is not a
        very large number, it is an undefined one, and the brains treat None
        as "unmeasured" throughout.
        """
        if self.declines == 0:
            return None
        return self.advances / self.declines


def pre_open_key(index_symbol: str) -> str:
    key = PRE_OPEN_KEYS.get(index_symbol.strip().upper())
    if key is None:
        raise DataAdapterError(
            f"NSE publishes no pre-open board for {index_symbol!r}. "
            f"Known: {sorted(set(PRE_OPEN_KEYS))}"
        )
    return key


def _decimal(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _int(value: Any) -> int:
    if value is None or value == "":
        return 0
    try:
        return int(Decimal(str(value)))
    except (InvalidOperation, ValueError):
        return 0


class NsePreOpenAdapter(ConstituentDataAdapter):
    """Constituent breadth from the pre-open auction board.

    Composes an already-configured `NsePublicAdapter` rather than opening its
    own session: the endpoint needs the same cookie warm-up as the rest of
    NSE's API, and two sessions would warm up twice and expire independently.
    """

    def __init__(
        self,
        source: Any,
        *,
        max_staleness: timedelta = DEFAULT_MAX_STALENESS,
        now: Any = None,
    ) -> None:
        """
        `now` is injectable because every staleness rule is untestable
        otherwise, and this one decides whether a whole brain reports a
        measurement or reports nothing.
        """
        self._source = source
        self._max_staleness = max_staleness
        self._now = now or (lambda: datetime.now(UTC))
        self._cache: dict[str, tuple[PreOpenSummary, list[dict[str, Any]]]] = {}

    async def _payload(self, index_symbol: str) -> tuple[PreOpenSummary, list[dict[str, Any]]]:
        key = pre_open_key(index_symbol)
        payload = await self._source._get_json(PRE_OPEN_URL, {"key": key})
        if not isinstance(payload, dict):
            raise DataAdapterError("NSE pre-open payload is not an object")
        rows = payload.get("data")
        if not isinstance(rows, list) or not rows:
            raise DataAdapterError(f"NSE published an empty pre-open board for {key}")

        raw_timestamp = payload.get("timestamp")
        if not isinstance(raw_timestamp, str):
            raise DataAdapterError("NSE pre-open payload has no timestamp")
        moment = parse_ist_timestamp(raw_timestamp)

        status = payload.get("niftyPreopenStatus") or {}
        summary = PreOpenSummary(
            timestamp=moment,
            advances=_int(payload.get("advances")),
            declines=_int(payload.get("declines")),
            unchanged=_int(payload.get("unchanged")),
            indicative_open=_decimal(status.get("lastPrice")),
            indicative_change=_decimal(status.get("change")),
            indicative_change_pct=(
                float(pct) if (pct := _decimal(status.get("pChange"))) is not None else None
            ),
            total_traded_value=_decimal(payload.get("totalTradedValue")),
        )
        resolved = (summary, rows)
        self._cache[key] = resolved
        return resolved

    def _is_stale(self, summary: PreOpenSummary) -> bool:
        return self._now() - summary.timestamp > self._max_staleness

    async def get_pre_open_summary(self, index_symbol: str) -> PreOpenSummary:
        """The exchange's own auction totals, stale or not.

        Deliberately not staleness-gated: a caller asking for the summary is
        asking about the auction, and the timestamp on it says when the
        auction was. It is `get_constituent_quotes` that must refuse to pass
        a stale snapshot off as the current market.
        """
        summary, _ = await self._payload(index_symbol)
        return summary

    async def get_constituents(self, index_symbol: str) -> list[ConstituentSpec]:
        """The index members and their capitalisation weights.

        Weights are **percentage points** — HDFCBANK comes back as 9.82, not
        0.0982 — matching `ConstituentSpec.weight` and what the Constituent
        brain divides by 100. Returning fractions here scaled every
        contribution down by 100 and made a +0.35% index move read as
        +0.0035%: small enough to look like a flat market rather than a bug.

        They are normalised over the constituents the board actually
        returned, so they sum to 100 across a partial board rather than to
        something less — a weight that silently shrinks would understate
        every contribution computed from it.
        """
        _, rows = await self._payload(index_symbol)
        caps: list[tuple[str, Decimal]] = []
        for row in rows:
            meta = row.get("metadata") or {}
            symbol = meta.get("symbol")
            cap = _decimal(meta.get("marketCap"))
            if not isinstance(symbol, str) or cap is None or cap <= 0:
                continue
            caps.append((symbol, cap))

        total = sum(cap for _, cap in caps)
        if total <= 0:
            raise DataAdapterError(
                f"NSE pre-open board for {index_symbol} carries no usable "
                "market capitalisation, so no weight can be derived"
            )
        return [
            ConstituentSpec(
                symbol=symbol,
                name=symbol,
                index_symbol=index_symbol.upper(),
                sector=UNKNOWN_SECTOR,
                weight=cap / total * 100,
            )
            for symbol, cap in caps
        ]

    async def get_constituent_quotes(
        self, symbols: list[str]
    ) -> list[ConstituentQuote]:
        """Auction prices for `symbols`, or nothing once the board is stale.

        The board is a snapshot of one auction. Serving it hours later would
        put a confident breadth reading behind an afternoon decision that
        nothing measured, so a stale board yields an empty list and the
        Constituent brain reports the absence rather than the snapshot.
        """
        if not symbols:
            return []
        wanted = {s.upper() for s in symbols}

        # Every cached board is searched: the caller passes symbols, not an
        # index, and a BANKNIFTY member is on the NIFTY board too.
        boards = list(self._cache.values())
        if not boards:
            for candidate in dict.fromkeys(PRE_OPEN_KEYS.values()):
                try:
                    boards.append(await self._payload(candidate))
                except DataAdapterError:
                    continue

        quotes: list[ConstituentQuote] = []
        seen: set[str] = set()
        for summary, rows in boards:
            if self._is_stale(summary):
                continue
            for row in rows:
                meta = row.get("metadata") or {}
                symbol = meta.get("symbol")
                if not isinstance(symbol, str) or symbol.upper() not in wanted:
                    continue
                if symbol.upper() in seen:
                    continue
                price = _decimal(meta.get("iep")) or _decimal(meta.get("lastPrice"))
                previous = _decimal(meta.get("previousClose"))
                if price is None or previous is None or price <= 0:
                    # A constituent that did not match has no price. Skipping
                    # it keeps the breadth count honest; a zero would read as
                    # a 100% decline.
                    continue
                seen.add(symbol.upper())
                quotes.append(
                    ConstituentQuote(
                        symbol=symbol,
                        timestamp=summary.timestamp,
                        ltp=price,
                        # An auction clears at a single price. open == high ==
                        # low == the IEP is the literal shape of the session.
                        open=price,
                        high=price,
                        low=price,
                        previous_close=previous,
                        volume=_int(meta.get("finalQuantity")),
                    )
                )
        return quotes
