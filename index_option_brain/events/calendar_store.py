"""The dated events an agent proposes and a human confirms.

Spec §4 names four trigger types that are calendar facts rather than
measurements — RBI policy, the Union Budget, an index rebalance, a
scheduled economic release. `ScheduledEventCalendar` has had no
implementation because no free Indian source serves them, and inventing
the dates was worse than not having them.

An agent reading circulars and press releases is the right way to fill
that gap. This module is what its output lands in.

Why a proposal cannot move the market decision
----------------------------------------------
An unverified date is dangerous in **both** directions. Treated as real
it makes the Risk Engine refuse to trade on a day nothing is happening;
missed, it lets the system trade through a day something is. There is no
conservative default, because neither direction of a *wrong* date is the
safe one — which is exactly the reason the interface shipped empty rather
than guessing.

So the state machine is the safety design, not bookkeeping.
`StoredEventCalendar` returns `CONFIRMED` rows and nothing else. A
proposal is a message to the operator; it changes nothing until they
answer. That is the same rule `LearningEngine` already follows: agent
output is an artifact a human promotes, never a live write.

Provenance is required
----------------------
`source` and `proposed_by` are stored because the first question about a
bad blackout is where the date came from, and a date whose provenance is
unrecorded cannot be re-checked. An agent proposal carries the URL it read.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from index_option_brain.contracts.risk import ScheduledEvent
from index_option_brain.database.engine import Database
from index_option_brain.database.models import CalendarEntryRow


class EntryState(StrEnum):
    PROPOSED = "PROPOSED"
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"


class EntryKind(StrEnum):
    RBI = "RBI"
    BUDGET = "BUDGET"
    REBALANCE = "REBALANCE"
    ECONOMIC = "ECONOMIC"
    EXPIRY = "EXPIRY"
    OTHER = "OTHER"


def _aware(value: datetime | None) -> datetime | None:
    """SQLite has no timezone type, so a `timezone=True` column reads back
    naive there. Everything is written as UTC, so re-attaching UTC restores
    the instant rather than guessing at one."""
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=UTC)


@dataclass(frozen=True)
class CalendarEntry:
    seq: int
    name: str
    starts_at: datetime
    kind: EntryKind
    state: EntryState
    blocks_new_entries: bool
    proposed_by: str
    proposed_at: datetime
    source: str | None = None
    decided_at: datetime | None = None
    decided_by: str | None = None
    note: str | None = None

    def as_scheduled_event(self) -> ScheduledEvent:
        return ScheduledEvent(
            name=self.name,
            starts_at=self.starts_at,
            blocks_new_entries=self.blocks_new_entries,
        )

    def describe(self) -> str:
        """One line, for a chat message or a log.

        The source is included because a date without provenance is one
        nobody can check, and confirming a date you cannot check is the
        failure this whole table exists to avoid.
        """
        when = self.starts_at.strftime("%a %d %b %Y %H:%M UTC")
        line = f"#{self.seq}  {when}  {self.kind}  {self.name}"
        if self.source:
            line += f"\n     source: {self.source}"
        if self.note:
            line += f"\n     note: {self.note}"
        return line


class CalendarStore:
    def __init__(self, database: Database) -> None:
        self._database = database

    def _to_entry(self, row: CalendarEntryRow) -> CalendarEntry:
        starts = _aware(row.starts_at)
        proposed = _aware(row.proposed_at)
        assert starts is not None and proposed is not None
        return CalendarEntry(
            seq=row.seq,
            name=row.name,
            starts_at=starts,
            kind=EntryKind(row.kind),
            state=EntryState(row.state),
            blocks_new_entries=row.blocks_new_entries,
            proposed_by=row.proposed_by,
            proposed_at=proposed,
            source=row.source,
            decided_at=_aware(row.decided_at),
            decided_by=row.decided_by,
            note=row.note,
        )

    async def propose(
        self,
        *,
        name: str,
        starts_at: datetime,
        kind: EntryKind = EntryKind.OTHER,
        proposed_by: str = "operator",
        source: str | None = None,
        note: str | None = None,
        blocks_new_entries: bool = True,
        now: datetime | None = None,
    ) -> CalendarEntry | None:
        """Record a proposed event, or None if it is already known.

        Idempotent on `(name, starts_at)` by a unique constraint, so a
        proposer re-run weekly does not accumulate copies of the same MPC
        meeting — and a date already confirmed or rejected is not quietly
        reset to PROPOSED by the next run.
        """
        moment = now or datetime.now(UTC)
        try:
            async with self._database.session() as session:
                row = CalendarEntryRow(
                    name=name.strip()[:120],
                    starts_at=starts_at.astimezone(UTC)
                    if starts_at.tzinfo
                    else starts_at.replace(tzinfo=UTC),
                    kind=str(kind),
                    state=str(EntryState.PROPOSED),
                    blocks_new_entries=blocks_new_entries,
                    source=source,
                    proposed_by=proposed_by[:64],
                    proposed_at=moment,
                    note=note[:280] if note else None,
                )
                session.add(row)
                await session.flush()
                return self._to_entry(row)
        except IntegrityError:
            return None

    async def decide(
        self,
        seq: int,
        *,
        state: EntryState,
        decided_by: str,
        now: datetime | None = None,
    ) -> CalendarEntry | None:
        """Confirm or reject one proposal.

        A decision is not reversible through this method by design: an
        entry that has already been decided is returned unchanged. Undoing
        a confirmation is a deliberate act at the machine, because a
        blackout appearing and disappearing from chat is a decision nobody
        can audit afterwards.
        """
        if state is EntryState.PROPOSED:
            raise ValueError("decide() sets CONFIRMED or REJECTED, not PROPOSED")
        async with self._database.session() as session:
            row = await session.get(CalendarEntryRow, seq)
            if row is None:
                return None
            if row.state != str(EntryState.PROPOSED):
                return self._to_entry(row)
            row.state = str(state)
            row.decided_at = now or datetime.now(UTC)
            row.decided_by = decided_by[:64]
            return self._to_entry(row)

    async def pending(self, limit: int = 20) -> list[CalendarEntry]:
        statement = (
            select(CalendarEntryRow)
            .where(CalendarEntryRow.state == str(EntryState.PROPOSED))
            .order_by(CalendarEntryRow.starts_at.asc())
            .limit(limit)
        )
        async with self._database.session() as session:
            rows = (await session.execute(statement)).scalars().all()
        return [self._to_entry(row) for row in rows]

    async def confirmed_between(
        self, start: datetime, end: datetime
    ) -> list[CalendarEntry]:
        statement = (
            select(CalendarEntryRow)
            .where(
                CalendarEntryRow.state == str(EntryState.CONFIRMED),
                CalendarEntryRow.starts_at >= start,
                CalendarEntryRow.starts_at <= end,
            )
            .order_by(CalendarEntryRow.starts_at.asc())
        )
        async with self._database.session() as session:
            rows = (await session.execute(statement)).scalars().all()
        return [self._to_entry(row) for row in rows]

    async def upcoming(self, *, limit: int = 20, now: datetime | None = None) -> list[CalendarEntry]:
        """Confirmed events still ahead — what the operator sees as live."""
        moment = now or datetime.now(UTC)
        statement = (
            select(CalendarEntryRow)
            .where(
                CalendarEntryRow.state == str(EntryState.CONFIRMED),
                CalendarEntryRow.starts_at >= moment,
            )
            .order_by(CalendarEntryRow.starts_at.asc())
            .limit(limit)
        )
        async with self._database.session() as session:
            rows = (await session.execute(statement)).scalars().all()
        return [self._to_entry(row) for row in rows]
