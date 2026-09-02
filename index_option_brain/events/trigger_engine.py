"""Spec §4 — the Event / Trigger Engine.

Compares two consecutive `MarketState` snapshots and reports what changed.

The contract invariant, which holds however this is implemented: **a trigger
only means "something changed; analyze it"**. It must never create an order,
and it must pass the significance filter before waking the full pipeline. So
this module produces `Event` objects and nothing else — it has no access to a
broker, no access to risk, and no way to reach either.

Why an event engine rather than a timer
---------------------------------------
A timer re-analyses a market that has not moved and misses one that moved
between ticks. The point of detection is that the pipeline runs when there is
something to reason about. That only pays off if the engine is quiet when the
market is: a detector that fires on every tick is worse than a timer, because
it costs the same and hides the signal. Hence the weight gate on constituent
moves, the OI floor before a ratio counts, the median rather than worst
spread, and the "fire on the crossing, not inside the state" shape of the
level, expiry and opening-range detectors.

Detection is also fail-soft. One detector raising must not lose the events the
others found: a chain with a malformed leg should not cost you the session
boundary. Failures are surfaced as an `EXCEPTIONAL_MARKET_EVENT` carrying the
error, because a detector that cannot read the market is itself news.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterable
from datetime import datetime

from index_option_brain.contracts.enums import TriggerType
from index_option_brain.contracts.events import Event
from index_option_brain.contracts.market_state import MarketState
from index_option_brain.contracts.risk import ScheduledEvent
from index_option_brain.events.config import TriggerEngineConfig
from index_option_brain.events.detectors import (
    CALENDAR_ONLY_TRIGGERS,
    CONSTITUENT_DETECTORS,
    MARKET_DETECTORS,
    OPTION_DETECTORS,
    TIME_DETECTORS,
    Detector,
    make_heartbeat_detector,
)


class ScheduledEventCalendar(ABC):
    """A source of known future events — RBI policy, the Budget, a rebalance.

    An interface with no implementation in this repository, deliberately. Four
    trigger types in spec §4 are calendar facts rather than measurements, and
    no free Indian source for that calendar was found. Fabricating the dates
    would put invented event risk into the Risk Engine's blackout logic, which
    is worse than not having them: the system would refuse to trade on days
    nothing was happening, and trade through days something was.
    """

    @abstractmethod
    def events_between(
        self, start: datetime, end: datetime
    ) -> list[ScheduledEvent]: ...


_CALENDAR_TRIGGER_BY_KEYWORD: tuple[tuple[str, TriggerType], ...] = (
    ("rbi", TriggerType.RBI_EVENT),
    ("monetary policy", TriggerType.RBI_EVENT),
    ("mpc", TriggerType.RBI_EVENT),
    ("budget", TriggerType.BUDGET_EVENT_RISK),
    ("rebalance", TriggerType.INDEX_REBALANCE),
    ("reconstitution", TriggerType.INDEX_REBALANCE),
)


class TriggerEngine(ABC):
    @abstractmethod
    def detect(
        self, previous_state: MarketState | None, current_state: MarketState
    ) -> list[Event]:
        """Compare two consecutive snapshots and return the events that fired.

        `previous_state` is None on the first tick, where only time triggers
        can fire — there is nothing to compare, and a system reporting
        "significant price movement" on its own arrival is reporting its own
        startup.
        """
        ...


class DeterministicTriggerEngine(TriggerEngine):
    """The production engine: a fixed set of detectors, run in order.

    Holds one piece of mutable state — the last heartbeat — and takes its
    cadence from snapshot timestamps rather than a timer, so a backtest
    produces the same heartbeats as a live session over the same data.
    """

    def __init__(
        self,
        config: TriggerEngineConfig | None = None,
        *,
        calendar: ScheduledEventCalendar | None = None,
        detectors: Iterable[Detector] | None = None,
    ) -> None:
        self._config = config or TriggerEngineConfig()
        self._calendar = calendar
        self._last_beat: dict[str, datetime] = {}
        self._detectors: tuple[Detector, ...] = tuple(
            detectors
            if detectors is not None
            else (
                *TIME_DETECTORS,
                *MARKET_DETECTORS,
                *OPTION_DETECTORS,
                *CONSTITUENT_DETECTORS,
                make_heartbeat_detector(self._last_beat),
            )
        )

    @property
    def unreachable_triggers(self) -> frozenset[TriggerType]:
        """Trigger types this engine cannot produce, given how it is wired.

        Exposed so the gap is checkable rather than a matter of reading the
        code. With no calendar attached, the four calendar-only triggers are
        unreachable and the system should say so instead of quietly never
        firing them.
        """
        if self._calendar is not None:
            return frozenset()
        return CALENDAR_ONLY_TRIGGERS

    def detect(
        self, previous_state: MarketState | None, current_state: MarketState
    ) -> list[Event]:
        events: list[Event] = []
        for detector in self._detectors:
            try:
                events.extend(detector(previous_state, current_state, self._config))
            except Exception as exc:  # noqa: BLE001 - one detector must not lose the rest
                events.append(self._detector_failure(detector, current_state, exc))
        events.extend(self._calendar_events(previous_state, current_state))
        return events

    def _detector_failure(
        self, detector: Detector, state: MarketState, exc: Exception
    ) -> Event:
        """A detector that cannot read the market is itself news.

        Reported rather than swallowed, and as EXCEPTIONAL_MARKET_EVENT rather
        than logged quietly, so the pipeline still wakes: the state that broke
        a detector is exactly the state worth looking at.
        """
        return Event(
            event_id=uuid.uuid4().hex[:12],
            trigger_type=TriggerType.EXCEPTIONAL_MARKET_EVENT,
            timestamp=state.timestamp,
            payload={
                "state_id": state.state_id,
                "detector": getattr(detector, "__name__", repr(detector)),
                "error": f"{type(exc).__name__}: {exc}",
                "reason": "a detector failed to read this state",
            },
            significance_score=0.8,
        )

    def _calendar_events(
        self, previous_state: MarketState | None, current_state: MarketState
    ) -> list[Event]:
        """Scheduled events that have come into view since the last snapshot.

        Empty without a calendar. The blackout window is the Risk Engine's
        (`RiskLimits.event_blackout_hours`); this only announces that the
        event is now near enough to matter.
        """
        if self._calendar is None:
            return []
        start = previous_state.timestamp if previous_state else current_state.timestamp
        horizon = current_state.timestamp
        scheduled = self._calendar.events_between(start, horizon)
        events: list[Event] = []
        for entry in scheduled:
            events.append(
                Event(
                    event_id=uuid.uuid4().hex[:12],
                    trigger_type=self._classify(entry),
                    timestamp=current_state.timestamp,
                    payload={
                        "state_id": current_state.state_id,
                        "name": entry.name,
                        "starts_at": entry.starts_at.isoformat(),
                        "blocks_new_entries": entry.blocks_new_entries,
                    },
                    significance_score=0.95 if entry.blocks_new_entries else 0.6,
                )
            )
        return events

    def _classify(self, entry: ScheduledEvent) -> TriggerType:
        name = entry.name.lower()
        for keyword, trigger in _CALENDAR_TRIGGER_BY_KEYWORD:
            if keyword in name:
                return trigger
        return TriggerType.MAJOR_SCHEDULED_ECONOMIC_EVENT
