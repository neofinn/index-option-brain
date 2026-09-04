"""Spec §4 — which events are worth waking the Quantitative Brain for.

Two jobs, and the second is the one that matters.

**A score floor.** Detectors set a significance from the magnitude of what
they measured; anything below the floor is recorded but does not trigger
analysis.

**A cooldown.** Without it the engine is a very expensive timer: a market
grinding through a level fires SUPPORT_RESISTANCE_TEST on every tick, and the
pipeline never gets to finish being useful before it is asked again. The
cooldown is per trigger type, so a price move being suppressed does not
suppress an IV collapse arriving in the same second.

Some triggers bypass both, and that is not a convenience. A session boundary
or a policy announcement changes what every other reading means; suppressing
one because something similar fired a minute ago would suppress the most
important wake-up of the day.

The filter takes its time from the events themselves, never from a clock, so
a replay suppresses exactly what the live session suppressed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

from index_option_brain.contracts.enums import TriggerType
from index_option_brain.contracts.events import Event
from index_option_brain.events.config import SignificanceFilterConfig


class SignificanceFilter(ABC):
    @abstractmethod
    def is_significant(self, event: Event) -> bool: ...


@dataclass(frozen=True)
class FilterDecision:
    """Why one event was let through or held back.

    Kept as a return value rather than a log line because "why did the system
    not react to that" is the question this layer gets asked, and a boolean
    cannot answer it.
    """

    event: Event
    significant: bool
    reason: str


@dataclass
class ThresholdSignificanceFilter(SignificanceFilter):
    """Score floor plus per-trigger cooldown."""

    config: SignificanceFilterConfig = field(default_factory=SignificanceFilterConfig)
    _last_passed: dict[TriggerType, datetime] = field(default_factory=dict)

    def is_significant(self, event: Event) -> bool:
        return self.evaluate(event).significant

    def evaluate(self, event: Event) -> FilterDecision:
        trigger = event.trigger_type

        if str(trigger) in self.config.always_significant:
            # Recorded so a later cooldown check has a reference, but never
            # itself suppressed.
            self._last_passed[trigger] = event.timestamp
            return FilterDecision(
                event, True, f"{trigger} always wakes the pipeline"
            )

        score = event.significance_score
        if score is None:
            # A detector that reported no magnitude has said nothing about how
            # much this matters. Treating that as significant would let an
            # unscored detector wake the pipeline forever.
            return FilterDecision(
                event, False, f"{trigger} carries no significance score"
            )
        if score < self.config.min_score:
            return FilterDecision(
                event,
                False,
                f"{trigger} scored {score:.2f}, below the "
                f"{self.config.min_score:.2f} floor",
            )

        cooldown = self.config.cooldown_seconds.get(
            str(trigger), self.config.default_cooldown_seconds
        )
        last = self._last_passed.get(trigger)
        if last is not None and cooldown > 0:
            elapsed = (event.timestamp - last).total_seconds()
            if 0 <= elapsed < cooldown:
                return FilterDecision(
                    event,
                    False,
                    f"{trigger} fired {elapsed:.0f}s ago, inside its "
                    f"{cooldown:.0f}s cooldown",
                )

        self._last_passed[trigger] = event.timestamp
        return FilterDecision(event, True, f"{trigger} scored {score:.2f}")

    def filter(self, events: list[Event]) -> list[Event]:
        """The significant events, in the order they were detected."""
        return [event for event in events if self.is_significant(event)]

    def explain(self, events: list[Event]) -> list[FilterDecision]:
        """Every event with its verdict, for the events that were held back."""
        return [self.evaluate(event) for event in events]

    def reset(self) -> None:
        """Clear the cooldown memory — used at a session boundary and between
        backtest runs, so one day's suppression does not carry into the next."""
        self._last_passed.clear()
