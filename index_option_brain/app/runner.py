"""The continuous engine loop.

Without this the system only thinks when someone opens the console, which
breaks the one thing the bar aggregator needs: NSE serves no history, so bars
exist only if snapshots keep arriving. A process that polls on page load
would still have no daily bars after a month.

So this is the piece that makes the engine *run* rather than be invoked. Each
cycle:

1. read a fresh `MarketState` from the live adapters,
2. feed it to the Trigger Engine against the previous state,
3. put the events through the significance filter,
4. run the Quantitative Brain only when something significant fired.

Step 4 is the point of having an event engine at all. Re-running a full
analysis every fifteen seconds on a market that has not moved is the
behaviour the §4 trigger contract exists to avoid.

What it cannot do
-----------------
Trade. There is no broker adapter, no account, and no order path in this
class. The Risk Engine is not even reached, because it requires an account
and a portfolio the system cannot see. That is not a limitation to be worked
around — a loop that authorized sizes against an invented balance would be
the most dangerous thing in the repository.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from index_option_brain.app.live import FeedUnavailable, LiveEngine
from index_option_brain.brain.pipeline import BrainCycleResult
from index_option_brain.contracts.enums import MarketSessionState
from index_option_brain.contracts.events import Event
from index_option_brain.contracts.market_state import MarketState
from index_option_brain.events import (
    DeterministicTriggerEngine,
    SignificanceFilterConfig,
    ThresholdSignificanceFilter,
    TriggerEngineConfig,
)


@dataclass(frozen=True)
class PollerConfig:
    """How often to look, and how hard to try when it goes wrong."""

    active_interval_seconds: float = 20.0
    """During the session. NSE's own timestamp advances roughly once a minute,
    so polling much faster buys nothing and only spends the rate limit."""
    closed_interval_seconds: float = 300.0
    """Outside it. A closed market answers with the same snapshot every time,
    and the aggregator discards it as non-advancing anyway."""
    max_backoff_seconds: float = 300.0
    backoff_multiplier: float = 2.0
    history_size: int = 200
    """How many recent events and cycles to keep for the console."""
    persist_every_cycles: int = 20
    """Snapshot observed bars this often.

    Not only on shutdown, because a crash is not a shutdown: periodic
    snapshots mean a kill -9 costs the bars since the last one rather than
    the whole session.
    """
    analyse_on_every_cycle: bool = False
    """Run the brain whether or not anything significant fired.

    Off by default: that is what makes this an event-driven engine rather than
    a timer with extra steps. Worth turning on only to debug the analysis
    chain itself.
    """


@dataclass
class PollerStats:
    """What the loop has actually been doing.

    Every field is counted, not estimated. The console shows these so an
    operator can tell "running and quiet" from "running and broken", which
    look identical from a single snapshot.
    """

    started_at: datetime | None = None
    cycles: int = 0
    successful_cycles: int = 0
    failed_cycles: int = 0
    consecutive_failures: int = 0
    events_detected: int = 0
    events_significant: int = 0
    analyses_run: int = 0
    last_poll_at: datetime | None = None
    last_success_at: datetime | None = None
    last_error: str | None = None
    current_interval_seconds: float = 0.0

    @property
    def uptime_seconds(self) -> float | None:
        if self.started_at is None:
            return None
        return (datetime.now(UTC) - self.started_at).total_seconds()

    @property
    def healthy(self) -> bool:
        """Running and getting data.

        A loop that is alive but has failed its last three polls is not
        healthy, and reporting it as such is how a dead feed goes unnoticed
        for a session.
        """
        return self.started_at is not None and self.consecutive_failures < 3


@dataclass
class MarketPoller:
    """Polls one or more symbols on a loop and reacts to what changed."""

    engine: LiveEngine
    symbols: tuple[str, ...] = ("NIFTY",)
    config: PollerConfig = field(default_factory=PollerConfig)
    trigger_config: TriggerEngineConfig = field(default_factory=TriggerEngineConfig)
    filter_config: SignificanceFilterConfig = field(
        default_factory=SignificanceFilterConfig
    )

    _task: asyncio.Task[None] | None = None
    _stop: asyncio.Event | None = None
    _previous: dict[str, MarketState] = field(default_factory=dict)
    _triggers: dict[str, DeterministicTriggerEngine] = field(default_factory=dict)
    _filters: dict[str, ThresholdSignificanceFilter] = field(default_factory=dict)
    _recent_events: deque[Event] = field(default_factory=lambda: deque(maxlen=200))
    _last_result: dict[str, BrainCycleResult] = field(default_factory=dict)
    stats: PollerStats = field(default_factory=PollerStats)

    # ------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stop = asyncio.Event()
        self.stats.started_at = datetime.now(UTC)
        self._task = asyncio.create_task(self._run(), name="market-poller")

    async def stop(self) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    # ----------------------------------------------------------------- loop

    async def _run(self) -> None:
        backoff = 0.0
        while self._stop is not None and not self._stop.is_set():
            session = await self._cycle_all()

            if self.stats.consecutive_failures:
                # Exponential backoff on a failing feed. Hammering an
                # unauthenticated public endpoint that is already refusing is
                # how a temporary block becomes a permanent one.
                backoff = min(
                    self.config.max_backoff_seconds,
                    max(self.config.active_interval_seconds, backoff)
                    * self.config.backoff_multiplier,
                )
                interval = backoff
            else:
                backoff = 0.0
                interval = (
                    self.config.active_interval_seconds
                    if session is not MarketSessionState.CLOSED
                    else self.config.closed_interval_seconds
                )

            self.stats.current_interval_seconds = interval
            with contextlib.suppress(TimeoutError, asyncio.TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=interval)

    async def _cycle_all(self) -> MarketSessionState:
        session = MarketSessionState.CLOSED
        for symbol in self.symbols:
            try:
                state = await self.cycle(symbol)
            except Exception as exc:  # noqa: BLE001 - the loop must survive anything
                self.stats.failed_cycles += 1
                self.stats.consecutive_failures += 1
                self.stats.last_error = f"{type(exc).__name__}: {exc}"
                continue
            if state is not None:
                session = state.session_state
        return session

    async def cycle(self, symbol: str) -> MarketState | None:
        """One poll of one symbol. Public so a test can drive it directly."""
        self.stats.cycles += 1
        self.stats.last_poll_at = datetime.now(UTC)

        try:
            state = await self.engine.market_state(symbol)
        except FeedUnavailable as exc:
            self.stats.failed_cycles += 1
            self.stats.consecutive_failures += 1
            self.stats.last_error = str(exc)
            return None

        self.stats.successful_cycles += 1
        self.stats.consecutive_failures = 0
        self.stats.last_error = None
        self.stats.last_success_at = datetime.now(UTC)

        previous = self._previous.get(symbol)
        trigger_engine = self._triggers.setdefault(
            symbol, DeterministicTriggerEngine(self.trigger_config)
        )
        significance = self._filters.setdefault(
            symbol, ThresholdSignificanceFilter(self.filter_config)
        )

        events = trigger_engine.detect(previous, state)
        self.stats.events_detected += len(events)
        significant = significance.filter(events)
        self.stats.events_significant += len(significant)
        for event in events:
            self._recent_events.append(event)

        # A new session clears the cooldown memory, so yesterday's suppression
        # does not carry into today.
        if (
            previous is not None
            and previous.session_state is MarketSessionState.CLOSED
            and state.session_state is not MarketSessionState.CLOSED
        ):
            significance.reset()

        if significant or self.config.analyse_on_every_cycle:
            self._last_result[symbol] = await self.engine.analysis(symbol)
            self.stats.analyses_run += 1

        self._previous[symbol] = state
        if (
            self.config.persist_every_cycles > 0
            and self.stats.successful_cycles % self.config.persist_every_cycles == 0
        ):
            try:
                self.engine.persist_bars()
            except OSError as exc:
                # Persistence failing must not stop the loop; losing bars is
                # worse than losing them slightly less often.
                self.stats.last_error = f"bar snapshot failed: {exc}"
        return state

    # -------------------------------------------------------------- reading

    def recent_events(self, limit: int = 50) -> list[Event]:
        """Newest first — an operator reads the top of the list."""
        return list(self._recent_events)[-limit:][::-1]

    def last_result(self, symbol: str) -> BrainCycleResult | None:
        return self._last_result.get(symbol.upper())

    def snapshot(self) -> dict[str, Any]:
        """The loop's own state, for the console's health panel."""
        return {
            "running": self.running,
            "healthy": self.stats.healthy,
            "symbols": list(self.symbols),
            "started_at": (
                self.stats.started_at.isoformat() if self.stats.started_at else None
            ),
            "uptime_seconds": self.stats.uptime_seconds,
            "cycles": self.stats.cycles,
            "successful_cycles": self.stats.successful_cycles,
            "failed_cycles": self.stats.failed_cycles,
            "consecutive_failures": self.stats.consecutive_failures,
            "events_detected": self.stats.events_detected,
            "events_significant": self.stats.events_significant,
            "analyses_run": self.stats.analyses_run,
            "last_poll_at": (
                self.stats.last_poll_at.isoformat() if self.stats.last_poll_at else None
            ),
            "last_success_at": (
                self.stats.last_success_at.isoformat()
                if self.stats.last_success_at
                else None
            ),
            "last_error": self.stats.last_error,
            "poll_interval_seconds": self.stats.current_interval_seconds,
            "unreachable_triggers": sorted(
                str(trigger)
                for engine in self._triggers.values()
                for trigger in engine.unreachable_triggers
            ),
        }
