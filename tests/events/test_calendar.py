"""The dated events an agent proposes and a human confirms."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest

from index_option_brain.database.engine import Database
from index_option_brain.events.calendar_store import (
    CalendarStore,
    EntryKind,
    EntryState,
)
from index_option_brain.events.stored_calendar import StoredEventCalendar

NOW = datetime(2026, 9, 10, 6, 0, tzinfo=UTC)


@pytest.fixture
async def database() -> AsyncIterator[Database]:
    db = Database.in_memory()
    await db.create_schema()
    yield db
    await db.aclose()


@pytest.fixture
def store(database: Database) -> CalendarStore:
    return CalendarStore(database)


class TestProposing:
    async def test_a_proposal_is_recorded_with_its_provenance(
        self, store: CalendarStore
    ) -> None:
        """The first question about a bad blackout is where the date came
        from, and a date whose provenance is unrecorded cannot be
        re-checked."""
        entry = await store.propose(
            name="RBI MPC",
            starts_at=NOW + timedelta(days=4),
            kind=EntryKind.RBI,
            proposed_by="calendar-agent",
            source="https://rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx?prid=1",
            now=NOW,
        )
        assert entry is not None
        assert entry.state is EntryState.PROPOSED
        assert entry.proposed_by == "calendar-agent"
        assert "rbi.org.in" in (entry.source or "")

    async def test_re_proposing_the_same_event_is_idempotent(
        self, store: CalendarStore
    ) -> None:
        """A proposer that runs weekly must not accumulate copies of the
        same MPC meeting."""
        when = NOW + timedelta(days=4)
        first = await store.propose(name="RBI MPC", starts_at=when, now=NOW)
        second = await store.propose(name="RBI MPC", starts_at=when, now=NOW)
        assert first is not None
        assert second is None
        assert len(await store.pending()) == 1

    async def test_a_decided_entry_is_not_reset_by_the_next_run(
        self, store: CalendarStore
    ) -> None:
        """The dangerous version of the idempotency bug: a proposer re-run
        quietly returning a confirmed date to PROPOSED would drop the
        blackout."""
        when = NOW + timedelta(days=4)
        entry = await store.propose(name="RBI MPC", starts_at=when, now=NOW)
        assert entry is not None
        await store.decide(entry.seq, state=EntryState.CONFIRMED, decided_by="me", now=NOW)
        assert await store.propose(name="RBI MPC", starts_at=when, now=NOW) is None
        assert len(await store.upcoming(now=NOW)) == 1

    async def test_a_naive_timestamp_is_read_as_utc(self, store: CalendarStore) -> None:
        # Naive on purpose: this is the case under test. A proposer parsing
        # a date out of a press release will produce one, and reading it as
        # local time would put the blackout five and a half hours out.
        naive = datetime(2027, 2, 1, 5, 30)  # noqa: DTZ001
        entry = await store.propose(name="Budget", starts_at=naive, now=NOW)
        assert entry is not None
        assert entry.starts_at == datetime(2027, 2, 1, 5, 30, tzinfo=UTC)


class TestDeciding:
    async def test_confirming_then_rejecting_does_not_reverse_it(
        self, store: CalendarStore
    ) -> None:
        entry = await store.propose(name="Budget", starts_at=NOW + timedelta(days=9), now=NOW)
        assert entry is not None
        await store.decide(entry.seq, state=EntryState.CONFIRMED, decided_by="me", now=NOW)
        again = await store.decide(
            entry.seq, state=EntryState.REJECTED, decided_by="me", now=NOW
        )
        assert again is not None
        assert again.state is EntryState.CONFIRMED

    async def test_deciding_records_who_and_when(self, store: CalendarStore) -> None:
        entry = await store.propose(name="Budget", starts_at=NOW + timedelta(days=9), now=NOW)
        assert entry is not None
        decided = await store.decide(
            entry.seq, state=EntryState.CONFIRMED, decided_by="satish", now=NOW
        )
        assert decided is not None
        assert decided.decided_by == "satish"
        assert decided.decided_at == NOW

    async def test_deciding_to_proposed_is_refused(self, store: CalendarStore) -> None:
        with pytest.raises(ValueError, match="CONFIRMED or REJECTED"):
            await store.decide(1, state=EntryState.PROPOSED, decided_by="me")

    async def test_a_missing_entry_returns_none(self, store: CalendarStore) -> None:
        assert await store.decide(404, state=EntryState.CONFIRMED, decided_by="me") is None


class TestOnlyConfirmedEventsReachTheRiskEngine:
    """The load-bearing rule.

    An unverified date is unsafe in both directions — believed it stops
    trading on a quiet day, missed it trades through a loud one. Since
    neither direction of an error is safe, an unverified date must not
    participate in the decision at all.
    """

    async def test_a_proposal_opens_no_blackout(self, store: CalendarStore) -> None:
        await store.propose(name="RBI MPC", starts_at=NOW + timedelta(hours=6), now=NOW)
        calendar = StoredEventCalendar(store)
        await calendar.refresh(now=NOW)
        assert calendar.events_between(NOW, NOW + timedelta(days=1)) == []

    async def test_a_confirmed_entry_does(self, store: CalendarStore) -> None:
        entry = await store.propose(
            name="RBI MPC", starts_at=NOW + timedelta(hours=6), now=NOW
        )
        assert entry is not None
        await store.decide(entry.seq, state=EntryState.CONFIRMED, decided_by="me", now=NOW)
        calendar = StoredEventCalendar(store)
        assert await calendar.refresh(now=NOW) == 1
        events = calendar.events_between(NOW, NOW + timedelta(days=1))
        assert [e.name for e in events] == ["RBI MPC"]
        assert events[0].blocks_new_entries is True

    async def test_a_rejected_entry_never_returns(self, store: CalendarStore) -> None:
        entry = await store.propose(name="Rumour", starts_at=NOW + timedelta(hours=6), now=NOW)
        assert entry is not None
        await store.decide(entry.seq, state=EntryState.REJECTED, decided_by="me", now=NOW)
        calendar = StoredEventCalendar(store)
        await calendar.refresh(now=NOW)
        assert calendar.events_between(NOW, NOW + timedelta(days=1)) == []

    async def test_the_window_is_respected(self, store: CalendarStore) -> None:
        far = await store.propose(name="Far", starts_at=NOW + timedelta(days=10), now=NOW)
        assert far is not None
        await store.decide(far.seq, state=EntryState.CONFIRMED, decided_by="me", now=NOW)
        calendar = StoredEventCalendar(store)
        await calendar.refresh(now=NOW)
        assert calendar.events_between(NOW, NOW + timedelta(days=2)) == []
        assert len(calendar.events_between(NOW, NOW + timedelta(days=14))) == 1


class TestStaleness:
    async def test_never_refreshed_counts_as_stale(self, store: CalendarStore) -> None:
        """An empty calendar and an unloaded one are indistinguishable from
        their contents, and only one of them means "nothing is
        scheduled"."""
        calendar = StoredEventCalendar(store)
        assert calendar.is_stale(now=NOW) is True
        assert calendar.age(now=NOW) is None
        assert "never loaded" in calendar.describe(now=NOW)

    async def test_a_fresh_snapshot_is_not_stale(self, store: CalendarStore) -> None:
        calendar = StoredEventCalendar(store)
        await calendar.refresh(now=NOW)
        assert calendar.is_stale(now=NOW) is False
        assert "fresh" in calendar.describe(now=NOW)

    async def test_an_old_snapshot_says_so(self, store: CalendarStore) -> None:
        """A calendar with two events that nobody refreshed for a week is
        worse than an empty one, because the empty one does not look
        answered."""
        calendar = StoredEventCalendar(store, max_age=timedelta(hours=1))
        await calendar.refresh(now=NOW)
        later = NOW + timedelta(hours=5)
        assert calendar.is_stale(now=later) is True
        assert "STALE" in calendar.describe(now=later)

    async def test_events_between_is_synchronous(self, store: CalendarStore) -> None:
        """The trigger engine that calls it is sync, so reading the database
        from inside it would block the event loop on every detection pass."""
        import inspect

        assert not inspect.iscoroutinefunction(StoredEventCalendar.events_between)
