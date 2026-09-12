"""The `ScheduledEventCalendar` the Risk Engine actually reads.

One rule, and it is the reason this file is separate from the store:
**only CONFIRMED entries are returned.** A proposal an agent made an hour
ago cannot open or close a blackout window. It is a message to the
operator until they answer it.

That is not caution for its own sake. The interface shipped with no
implementation precisely because a wrong date is unsafe in both
directions — believed, it stops trading on a quiet day; missed, it trades
through a loud one. Since neither direction of an error is the safe one,
an unverified date must not participate in the decision at all.

Synchronous by necessity
------------------------
`ScheduledEventCalendar.events_between` is sync, because the trigger
engine that calls it is. Reading the database from inside it would mean
blocking the event loop on every detection pass. So this class holds a
**snapshot** that something async refreshes on a schedule, and reports how
old that snapshot is — because a calendar nobody has refreshed since
Tuesday looks exactly like a calendar with nothing in it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Protocol, runtime_checkable

from index_option_brain.contracts.risk import ScheduledEvent
from index_option_brain.events.calendar_store import CalendarEntry, CalendarStore
from index_option_brain.events.trigger_engine import ScheduledEventCalendar

#: How far ahead to load. The Risk Engine's blackout window is measured in
#: hours, and the trigger engine looks days out, so a fortnight covers both
#: with room for a proposer that runs weekly.
DEFAULT_HORIZON = timedelta(days=14)

#: How often to reload. Short, because a confirmation from chat has to
#: take effect in minutes rather than at the next restart.
DEFAULT_REFRESH_INTERVAL = timedelta(minutes=10)

#: Past this the snapshot is reported stale. Deliberately a different and
#: much longer threshold than the refresh interval: one answers "should I
#: reload", the other answers "should anyone trust this". Collapsing them
#: into one number makes a calendar that is merely due a reload look
#: untrustworthy, and a calendar that has genuinely been unreachable for
#: hours look fine.
DEFAULT_MAX_AGE = timedelta(hours=6)


@runtime_checkable
class RefreshableCalendar(Protocol):
    """The half of `StoredEventCalendar` a caller needs to keep it current.

    A protocol rather than a method on `ScheduledEventCalendar`, because a
    calendar backed by a static file or a vendor API has nothing to
    refresh and should not be forced to pretend otherwise.
    """

    def needs_refresh(self, *, now: datetime | None = None) -> bool: ...

    async def refresh(self, *, now: datetime | None = None) -> int: ...


class StoredEventCalendar(ScheduledEventCalendar):
    """A calendar over confirmed rows, refreshed out of band."""

    def __init__(
        self,
        store: CalendarStore,
        *,
        horizon: timedelta = DEFAULT_HORIZON,
        max_age: timedelta = DEFAULT_MAX_AGE,
        refresh_interval: timedelta = DEFAULT_REFRESH_INTERVAL,
    ) -> None:
        self._store = store
        self._horizon = horizon
        self._max_age = max_age
        self._refresh_interval = refresh_interval
        self._entries: list[CalendarEntry] = []
        self._loaded_at: datetime | None = None

    async def refresh(self, *, now: datetime | None = None) -> int:
        """Reload the snapshot. Returns how many confirmed events it holds."""
        moment = now or datetime.now(UTC)
        self._entries = await self._store.confirmed_between(
            moment - timedelta(days=1), moment + self._horizon
        )
        self._loaded_at = moment
        return len(self._entries)

    @property
    def loaded_at(self) -> datetime | None:
        return self._loaded_at

    def age(self, *, now: datetime | None = None) -> timedelta | None:
        if self._loaded_at is None:
            return None
        return (now or datetime.now(UTC)) - self._loaded_at

    def is_stale(self, *, now: datetime | None = None) -> bool:
        """Never refreshed counts as stale.

        An empty calendar and an unloaded one are indistinguishable from
        their contents, and only one of them means "nothing is scheduled".
        """
        age = self.age(now=now)
        return age is None or age > self._max_age

    def needs_refresh(self, *, now: datetime | None = None) -> bool:
        """Due a reload — a much lower bar than being untrustworthy."""
        age = self.age(now=now)
        return age is None or age > self._refresh_interval

    def events_between(self, start: datetime, end: datetime) -> list[ScheduledEvent]:
        return [
            entry.as_scheduled_event()
            for entry in self._entries
            if start <= entry.starts_at <= end
        ]

    def describe(self, *, now: datetime | None = None) -> str:
        """A line for the console and the chat bot.

        Says the age, not just the count. A calendar with two events in it
        that nobody has refreshed for a week is worse than an empty one,
        because the empty one does not look answered.
        """
        if self._loaded_at is None:
            return "Calendar: never loaded — no confirmed events are in effect"
        age = self.age(now=now)
        minutes = int((age or timedelta()).total_seconds() // 60)
        state = "STALE" if self.is_stale(now=now) else "fresh"
        return (
            f"Calendar: {len(self._entries)} confirmed event(s), "
            f"loaded {minutes}m ago ({state})"
        )
