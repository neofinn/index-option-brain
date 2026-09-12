"""Where accepted alerts wait for the engine to pick them up.

The receiver and the analysis pipeline are deliberately not the same
process. A webhook has to answer TradingView in a couple of seconds or the
alert log fills with failures, and a full analysis cycle — chain fetch,
greeks, nine brains, risk, gate — does not fit in that budget. So the
receiver's whole job is: authenticate, validate, admit, return 200. The
engine drains at its own pace.

That makes the queue the interesting object, and it has two jobs.

**Suppress duplicates.** TradingView delivers at most once, but "at most
once" is not "exactly once at most once per bar": an alert can fire again
on the same bar across a reconnect, and an operator debugging a template
will re-fire by hand. `TradingViewAlert.dedupe_key` is keyed on the *bar*,
so both cases collapse to one event.

**Never grow without bound, and never lose a drop silently.** If nothing
drains, memory is the failure. The bound is a fixed capacity; when it is
reached the *oldest* alert is discarded, because in a market the newest
observation is the one worth keeping — and `dropped` counts every one, so
a console can show that the receiver is outrunning the engine instead of
that everything is fine.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from index_option_brain.contracts.events import Event
from index_option_brain.integrations.tradingview.alert import (
    AlertRejected,
    RejectionReason,
    TradingViewAlert,
)

DEFAULT_CAPACITY = 256
DEFAULT_DEDUPE_RETENTION = timedelta(hours=1)


class AlertInbox:
    """A bounded, duplicate-suppressing queue of chart events."""

    def __init__(
        self,
        *,
        capacity: int = DEFAULT_CAPACITY,
        dedupe_retention: timedelta = DEFAULT_DEDUPE_RETENTION,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._events: deque[Event] = deque()
        self._capacity = capacity
        self._retention = dedupe_retention
        self._clock = clock or (lambda: datetime.now(UTC))
        # Insertion-ordered, so expiry is a walk from the front and never a
        # scan of the whole map.
        self._seen: OrderedDict[str, datetime] = OrderedDict()
        self.dropped = 0
        self.duplicates = 0

    def __len__(self) -> int:
        return len(self._events)

    def _expire(self, now: datetime) -> None:
        cutoff = now - self._retention
        while self._seen:
            key, seen_at = next(iter(self._seen.items()))
            if seen_at > cutoff:
                break
            self._seen.popitem(last=False)
            del key

    def admit(self, alert: TradingViewAlert) -> Event:
        """Queue an alert's event, or raise if it is one already seen."""
        now = self._clock()
        self._expire(now)
        key = alert.dedupe_key
        if key in self._seen:
            self.duplicates += 1
            raise AlertRejected(
                RejectionReason.DUPLICATE,
                "this alert has already been accepted for this bar",
            )
        self._seen[key] = now
        event = alert.to_event()
        if len(self._events) >= self._capacity:
            self._events.popleft()
            self.dropped += 1
        self._events.append(event)
        return event

    def drain(self) -> list[Event]:
        """Take everything queued, oldest first, and leave the inbox empty.

        Draining does not clear the dedupe memory: an alert consumed a
        second ago is still a duplicate if it arrives again.
        """
        drained = list(self._events)
        self._events.clear()
        return drained

    def peek(self, limit: int = 20) -> list[Event]:
        """The most recent events, newest first, without consuming them."""
        return list(self._events)[-limit:][::-1]

    def stats(self) -> dict[str, Any]:
        return {
            "queued": len(self._events),
            "capacity": self._capacity,
            "dropped": self.dropped,
            "duplicates": self.duplicates,
        }
