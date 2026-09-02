"""The live engine behind the operations console.

One object owns the adapters, measures whether they are actually working, and
turns their output into the shapes the console renders. It exists so the
console has nothing to fall back on: there is no sample data path here, and a
provider that cannot answer produces an explicit unavailable state carrying
the reason.

Health is measured, never assumed
---------------------------------
`probe` calls the endpoints and times them. A capability appears in
`verified_capabilities` only after a call for it returned data. That is the
difference between what a `ProviderDescriptor` claims and what is working
right now, and an operator deciding whether to trade needs the second.

Everything is cached for a few seconds, for coherence rather than politeness:
the console makes several requests to paint one screen, and if they landed on
different snapshots the screen would describe a market that never existed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, TypeVar

from index_option_brain.brain.pipeline import BrainCycleResult, QuantitativeBrain
from index_option_brain.contracts.enums import BarInterval, MarketSessionState
from index_option_brain.contracts.market_state import MarketState
from index_option_brain.contracts.provider import (
    Capability,
    ProviderConnectionState,
    ProviderHealth,
)
from index_option_brain.data.adapters.base import DataAdapterError
from index_option_brain.data.adapters.nse_public import (
    NSE_PUBLIC_DESCRIPTOR,
    NsePublicAdapter,
    index_config_from_master,
)
from index_option_brain.data.bar_aggregator import AggregatingIndexAdapter
from index_option_brain.data.bar_store import BarStore
from index_option_brain.data.dhan_instruments import DhanInstrumentMaster
from index_option_brain.state.market_state_builder import (
    InMemoryIvHistoryStore,
    MarketStateBuilder,
)

DEFAULT_SYMBOLS = ("NIFTY", "BANKNIFTY")

T = TypeVar("T")


class FeedUnavailable(RuntimeError):
    """The feed could not answer. Carries the reason, for display."""


@dataclass
class _Cached:
    at: float
    value: Any


@dataclass
class LiveEngine:
    """Live market data and analysis for the console.

    Deliberately holds no positions, account or orders: nothing here can place
    a trade, and the console's execution panels stay empty until a broker
    adapter exists. An account figure invented for display would be the most
    dangerous fake number in the system.
    """

    cache_seconds: float = 5.0
    intraday_interval: BarInterval = BarInterval.MINUTE_5
    bar_store: BarStore | None = None
    """Makes observed bars survive a restart. None keeps them in memory only."""
    _cache: dict[str, _Cached] = field(default_factory=dict)
    _nse: NsePublicAdapter | None = None
    _index: AggregatingIndexAdapter | None = None
    _builder: MarketStateBuilder | None = None
    _brain: QuantitativeBrain = field(default_factory=QuantitativeBrain)
    _iv_history: InMemoryIvHistoryStore = field(
        default_factory=InMemoryIvHistoryStore
    )
    _health: dict[str, ProviderHealth] = field(default_factory=dict)
    _master: DhanInstrumentMaster | None = None

    # ------------------------------------------------------------ lifecycle

    async def load_instruments(self) -> DhanInstrumentMaster:
        """Fetch contract specifications from the exchange's own record.

        Called before the adapters are built, so lot size, tick size and
        strike step are read rather than assumed. Dhan publishes this without
        authentication, which is what makes it usable before any
        subscription — and it is the difference between a verified contract
        size and a constant with a warning comment on it.
        """
        if self._master is None:
            self._master = await DhanInstrumentMaster.load()
        return self._master

    async def ensure_ready(self) -> None:
        """Load contract specifications, then build the adapters around them.

        A failure here is not fatal: the adapters fall back to the snapshot in
        DEFAULT_INDEX_CONFIG, and `instrument_source` reports which was used
        so the console can say so. Refusing to start would make a transient
        CDN outage an availability outage.
        """
        try:
            master = await self.load_instruments()
        except DataAdapterError:
            self._ensure()
            return
        config = index_config_from_master(master)
        if config:
            self._nse = NsePublicAdapter(index_config=config)
            self._index = AggregatingIndexAdapter(
                self._nse,
                intervals=(self.intraday_interval, BarInterval.DAY),
                store=self.bar_store,
            )
            self._builder = MarketStateBuilder(
                self._index,
                None,
                self._nse,
                self._nse,
                self._iv_history,
                intraday_interval=self.intraday_interval,
            )
        else:
            self._ensure()

    @property
    def instrument_source(self) -> str:
        """Where the contract specifications in use came from.

        Surfaced because a stale lot size mis-sizes every order, and an
        operator needs to be able to tell a verified table from a fallback.
        """
        if self._master is not None:
            return "dhan_instrument_master"
        return "bundled_snapshot"

    def _ensure(self) -> tuple[NsePublicAdapter, AggregatingIndexAdapter, MarketStateBuilder]:
        if self._nse is None or self._index is None or self._builder is None:
            self._nse = NsePublicAdapter()
            self._index = AggregatingIndexAdapter(
                self._nse,
                intervals=(self.intraday_interval, BarInterval.DAY),
                store=self.bar_store,
            )
            self._builder = MarketStateBuilder(
                self._index,
                # No provider serves index breadth. Stated explicitly rather
                # than omitted, so the gap is visible in the wiring.
                None,
                self._nse,
                self._nse,
                self._iv_history,
                intraday_interval=self.intraday_interval,
            )
        return self._nse, self._index, self._builder

    def persist_bars(self) -> int:
        """Snapshot observed bars. Returns the number of series written."""
        if self._index is None:
            return 0
        return self._index.persist()

    async def aclose(self) -> None:
        # Snapshot before tearing down, so a clean shutdown keeps the session's
        # bars rather than discarding them.
        try:
            self.persist_bars()
        except OSError:
            # A failure to persist must not prevent a clean shutdown.
            pass
        if self._nse is not None:
            await self._nse.aclose()
        self._nse = None
        self._index = None
        self._builder = None
        self._cache.clear()

    # --------------------------------------------------------------- cache

    def _fresh(self, key: str, kind: type[T]) -> T | None:
        entry = self._cache.get(key)
        if entry is None or time.monotonic() - entry.at > self.cache_seconds:
            return None
        return entry.value if isinstance(entry.value, kind) else None

    def _store(self, key: str, value: T) -> T:
        self._cache[key] = _Cached(at=time.monotonic(), value=value)
        return value

    # -------------------------------------------------------------- health

    def health(self, provider_id: str) -> ProviderHealth:
        """The last measured health, or an explicitly unmeasured one.

        NOT_CONFIGURED rather than a zeroed CONNECTED: a console reporting 0 ms
        for a provider it has never called would be reporting a measurement it
        does not have.
        """
        return self._health.get(provider_id, ProviderHealth(provider_id=provider_id))

    def all_health(self) -> dict[str, ProviderHealth]:
        return dict(self._health)

    async def probe(self, symbol: str = "NIFTY") -> ProviderHealth:
        """Call the live endpoints and record what actually worked."""
        cached = self._fresh(f"probe:{symbol}", ProviderHealth)
        if cached is not None:
            return cached

        nse, _, _ = self._ensure()
        verified: set[Capability] = set()
        errors: list[str] = []
        started = time.perf_counter()

        try:
            await nse.get_index_quote(symbol)
            verified.add(Capability.INDEX_QUOTE)
        except DataAdapterError as exc:
            errors.append(f"index quote: {exc}")

        try:
            await nse.get_india_vix()
            verified.add(Capability.INDIA_VIX)
        except DataAdapterError as exc:
            errors.append(f"India VIX: {exc}")

        try:
            expiries = await nse.get_available_expiries(symbol)
            verified.add(Capability.EXPIRY_LIST)
        except DataAdapterError as exc:
            errors.append(f"expiries: {exc}")
            expiries = []

        if expiries:
            try:
                await nse.get_option_chain(symbol, expiries[0])
                verified.add(Capability.OPTION_CHAIN)
            except DataAdapterError as exc:
                errors.append(f"chain: {exc}")

        latency_ms = (time.perf_counter() - started) * 1000.0
        declared = NSE_PUBLIC_DESCRIPTOR.capabilities

        if not verified:
            state = ProviderConnectionState.FAILED
        elif verified >= set(declared):
            state = ProviderConnectionState.CONNECTED
        else:
            # Reachable but not serving everything it declared. This is a real
            # and useful state, not a euphemism for broken.
            state = ProviderConnectionState.DEGRADED

        now = datetime.now(UTC)
        health = ProviderHealth(
            provider_id=NSE_PUBLIC_DESCRIPTOR.provider_id,
            state=state,
            checked_at=now.isoformat(),
            latency_ms=round(latency_ms, 1),
            last_success_at=now.isoformat() if verified else None,
            last_error="; ".join(errors) or None,
            verified_capabilities=frozenset(verified),
        )
        self._health[health.provider_id] = health
        return self._store(f"probe:{symbol}", health)

    # -------------------------------------------------------- market state

    async def market_state(self, symbol: str) -> MarketState:
        cached = self._fresh(f"state:{symbol}", MarketState)
        if cached is not None:
            return cached
        _, _, builder = self._ensure()
        try:
            state = await builder.build(symbol)
        except DataAdapterError as exc:
            raise FeedUnavailable(str(exc)) from exc
        return self._store(f"state:{symbol}", state)

    async def analysis(self, symbol: str) -> BrainCycleResult:
        """Run the brain on the live state.

        No account or portfolio is passed, so the Risk Engine does not run and
        nothing here can present itself as authorized. That is correct until a
        broker is connected: authorizing a size against an account the system
        cannot see would be the worst possible invention.
        """
        cached = self._fresh(f"analysis:{symbol}", BrainCycleResult)
        if cached is not None:
            return cached
        state = await self.market_state(symbol)
        return self._store(f"analysis:{symbol}", self._brain.run(state))

    def bar_coverage(self, symbol: str) -> dict[str, Any]:
        """How much history has been observed, and whether it has holes.

        Surfaced because a short bar series and a gappy one look identical in
        a chart but mean different things to an indicator.
        """
        _, index, _ = self._ensure()
        report: dict[str, Any] = {}
        for interval in (self.intraday_interval, BarInterval.DAY):
            stats = index.stats(symbol, interval)
            report[str(interval)] = {
                "bars": len(index.aggregator(symbol, interval).completed),
                "observations": stats.observations,
                "missing_buckets": stats.missing_buckets,
                "discarded_partial": stats.bars_discarded_partial,
                "has_gaps": stats.has_gaps,
                "first_observation": (
                    stats.first_observation.isoformat()
                    if stats.first_observation
                    else None
                ),
                "seeded": stats.seeded_bars,
            }
        return report


def money(value: Decimal | float | None) -> float | None:
    """Convert for transport, or None. There is no zero default: a missing
    figure must not arrive at the console looking like a measured zero."""
    return None if value is None else float(value)


def session_label(state: MarketSessionState) -> str:
    return {
        MarketSessionState.PRE_MARKET: "Pre-market",
        MarketSessionState.OPENING: "Opening auction",
        MarketSessionState.ACTIVE: "Active",
        MarketSessionState.CLOSING: "Closing",
        MarketSessionState.CLOSED: "Closed",
    }[state]
