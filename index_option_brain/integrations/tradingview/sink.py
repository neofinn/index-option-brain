"""Getting an accepted alert out of the receiver and into the engine.

The receiver is a separate process, which is what keeps the console
read-only and keeps the tailnet private — and which also means the
in-memory inbox is invisible to everything else. So an accepted alert is
written to `system_events`, and the engine reads it from there.

Idempotence lives in the id, not in the memory
----------------------------------------------
`AlertInbox`'s dedupe set is a per-process optimisation: it stops the
obvious repeat cheaply and it is forgotten on restart. What actually makes
redelivery safe is that `Event.event_id` is a UUIDv5 over the alert's bar
key, so the same alert always derives the same id however many processes
have handled it. The consumer skips ids it has already seen, and a receiver
restart in the middle of a redelivery costs a duplicate row rather than a
duplicate wake.

A failed write is not a delivered alert
---------------------------------------
If a sink is configured and its write fails, the receiver answers 503. It
would be easy to answer 200 — the alert did reach the in-memory inbox — but
in a two-process deployment that inbox is a dead end, and a 200 tells
TradingView's alert log that everything worked while nothing downstream
will ever see it. A failed webhook in that log is the signal the operator
needs.

This is deliberately not the capture recorder's fail-soft rule. Capture
losing a snapshot costs one row of a corpus; a lost alert is the trigger
that was supposed to wake the analysis.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol

from sqlalchemy import select

from index_option_brain.contracts.enums import TriggerType
from index_option_brain.contracts.events import Event
from index_option_brain.database.engine import Database
from index_option_brain.database.models import SystemEventRow

#: The `system_events.kind` every chart alert is written under, so one
#: indexed query finds them and nothing else is mistaken for one.
ALERT_KIND = "tradingview_alert"


class AlertSink(Protocol):
    """Where an accepted alert goes after the receiver has validated it."""

    async def record(self, event: Event) -> None: ...


class NullAlertSink:
    """Keeps the alert in memory and nowhere else.

    Correct for a single-process deployment where the consumer holds the
    same `AlertInbox`, and correct for testing the receiver. Wrong for the
    two-process deployment this package recommends, which is why it has to
    be chosen explicitly rather than being what you get by leaving the sink
    unset.
    """

    async def record(self, event: Event) -> None:
        return None


class DatabaseAlertSink:
    """Writes accepted alerts to `system_events`.

    `severity` is "info" because an alert is an observation, not a fault.
    The event id is carried in `detail` rather than in a column of its own:
    `system_events` is a shared operational log and its shape is not this
    integration's to change, so the consumer dedupes on the value it finds
    there.
    """

    def __init__(self, database: Database) -> None:
        self._database = database

    async def record(self, event: Event) -> None:
        async with self._database.session() as session:
            session.add(
                SystemEventRow(
                    kind=ALERT_KIND,
                    severity="info",
                    message=f"{event.trigger_type} on {event.payload.get('symbol', '?')}",
                    detail={
                        "event_id": event.event_id,
                        "trigger_type": str(event.trigger_type),
                        **event.payload,
                    },
                    occurred_at=event.timestamp,
                )
            )


def _event_from_detail(detail: dict[str, Any], occurred_at: datetime) -> Event | None:
    """Rebuild an `Event` from a row, or None if the row is not one.

    A row whose detail is missing an id or carries an unknown trigger name
    is skipped rather than repaired. `system_events` is a shared table, and
    guessing at a malformed row here would put an invented trigger into the
    pipeline.
    """
    event_id = detail.get("event_id")
    raw_trigger = detail.get("trigger_type")
    if not isinstance(event_id, str) or not isinstance(raw_trigger, str):
        return None
    try:
        trigger = TriggerType(raw_trigger)
    except ValueError:
        return None
    payload = {k: v for k, v in detail.items() if k not in {"event_id", "trigger_type"}}
    # SQLite has no timezone type, so a column declared `timezone=True`
    # still reads back naive there. The value was written as UTC — the
    # alert's firing time is converted to UTC before it ever reaches the
    # sink — so re-attaching UTC restores the instant rather than guessing
    # at one. Without this a round trip through SQLite yields a naive
    # timestamp, and the first comparison against an aware `now`
    # downstream raises TypeError.
    timestamp = occurred_at if occurred_at.tzinfo else occurred_at.replace(tzinfo=UTC)
    return Event(
        event_id=event_id,
        trigger_type=trigger,
        timestamp=timestamp,
        payload=payload,
        significance_score=None,
    )


async def pending_alerts(
    database: Database, *, since: datetime | None = None, limit: int = 100
) -> list[Event]:
    """Chart alerts at or after `since`, oldest first, deduped by id.

    Deduping here rather than trusting the writer is the point: two
    receiver processes, or one that restarted mid-redelivery, can both have
    written the same alert, and the deterministic event id is what makes
    that recoverable.

    The cursor is inclusive — `>=`, not `>` — and that is deliberate.
    `occurred_at` is the alert's firing time, which has second resolution
    and which two alerts on the same bar close routinely share. With a
    strict `>`, a consumer that stores the last `occurred_at` it saw would
    skip its sibling permanently: the alert is in the table, the cursor has
    passed it, and nothing ever returns it. Inclusive costs re-reading the
    boundary row on the next poll, which the consumer's id-dedupe already
    absorbs — so the failure mode is a repeat rather than a loss. Consumers
    must therefore dedupe on `event_id` across polls, exactly as they must
    for redelivery.
    """
    statement = (
        select(SystemEventRow)
        .where(SystemEventRow.kind == ALERT_KIND)
        .order_by(SystemEventRow.occurred_at.asc())
        .limit(limit)
    )
    if since is not None:
        # Normalised because the stored column is UTC: an IST-aware cursor
        # compared against it directly would be five and a half hours off.
        cursor = since.astimezone(UTC) if since.tzinfo else since.replace(tzinfo=UTC)
        statement = statement.where(SystemEventRow.occurred_at >= cursor)

    async with database.session() as session:
        rows = (await session.execute(statement)).scalars().all()

    seen: set[str] = set()
    events: list[Event] = []
    for row in rows:
        event = _event_from_detail(dict(row.detail or {}), row.occurred_at)
        if event is None or event.event_id in seen:
            continue
        seen.add(event.event_id)
        events.append(event)
    return events
