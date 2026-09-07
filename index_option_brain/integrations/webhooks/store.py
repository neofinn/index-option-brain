"""Where deliveries live between the POST and the poll.

The store is the whole conversion. A webhook is push and an API is pull,
and the only thing standing between them is somewhere durable to keep a
payload plus a way to ask "what has arrived since I last looked".

The cursor is an integer
------------------------
`seq` is a monotonic autoincrementing key, so `since` is an exact `>` and a
poller that remembers the last value it saw can neither skip a delivery nor
see one twice. That is not the obvious choice — a received-at timestamp
looks like the natural cursor — and it is here because the timestamp
version was already tried elsewhere in this codebase and had to be made
inclusive to stop two deliveries in the same second from losing one
permanently, which then re-reads the boundary row on every single poll.

Retention is enforced on write
------------------------------
A sender misconfigured to fire every second fills a disk in a day. The cap
is applied on every insert rather than by a nightly job, because the
consequence of an unbounded table here lands on the engine sharing that
disk, not on the gateway — and a full disk stops the system from gating a
trade, which is the failure this project spends the most effort avoiding.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, func, select

from index_option_brain.database.engine import Database
from index_option_brain.database.models import WebhookDeliveryRow


@dataclass(frozen=True)
class Delivery:
    """One stored delivery, as the read API returns it."""

    seq: int
    endpoint: str
    received_at: datetime
    payload: dict[str, Any]
    source_ip: str | None = None
    content_type: str | None = None
    body_bytes: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "endpoint": self.endpoint,
            "received_at": self.received_at.isoformat(),
            "source_ip": self.source_ip,
            "content_type": self.content_type,
            "body_bytes": self.body_bytes,
            "payload": self.payload,
        }


@dataclass(frozen=True)
class EndpointStats:
    count: int
    last_seq: int | None
    last_received_at: datetime | None


def _aware(value: datetime | None) -> datetime | None:
    """SQLite has no timezone type, so a `timezone=True` column still reads
    back naive there. Everything is written as UTC, so re-attaching UTC
    restores the instant rather than guessing at one — and without this the
    first comparison against an aware `now` downstream raises TypeError."""
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=UTC)


class DeliveryStore:
    def __init__(self, database: Database) -> None:
        self._database = database

    def _to_delivery(self, row: WebhookDeliveryRow) -> Delivery:
        received = _aware(row.received_at)
        assert received is not None  # the column is NOT NULL
        return Delivery(
            seq=row.seq,
            endpoint=row.endpoint,
            received_at=received,
            payload=dict(row.payload or {}),
            source_ip=row.source_ip,
            content_type=row.content_type,
            body_bytes=row.body_bytes,
        )

    async def record(
        self,
        *,
        endpoint: str,
        payload: dict[str, Any],
        received_at: datetime,
        source_ip: str | None,
        content_type: str | None,
        body_bytes: int,
        retain: int,
    ) -> int:
        """Store one delivery and return its cursor value."""
        async with self._database.session() as session:
            row = WebhookDeliveryRow(
                endpoint=endpoint,
                received_at=received_at,
                source_ip=source_ip,
                content_type=content_type,
                payload=payload,
                body_bytes=body_bytes,
            )
            session.add(row)
            await session.flush()
            seq = row.seq

            # Keep the newest `retain` rows for this endpoint. The offset
            # query finds the oldest row that survives; everything below it
            # goes. One statement rather than a fetch-then-delete loop, so
            # a busy endpoint does not pull its own history into memory to
            # throw it away.
            cutoff = (
                select(WebhookDeliveryRow.seq)
                .where(WebhookDeliveryRow.endpoint == endpoint)
                .order_by(WebhookDeliveryRow.seq.desc())
                .limit(1)
                .offset(retain - 1)
            )
            oldest_kept = (await session.execute(cutoff)).scalar_one_or_none()
            if oldest_kept is not None:
                await session.execute(
                    delete(WebhookDeliveryRow).where(
                        WebhookDeliveryRow.endpoint == endpoint,
                        WebhookDeliveryRow.seq < oldest_kept,
                    )
                )
        return seq

    async def since(
        self, endpoint: str, *, cursor: int = 0, limit: int = 50
    ) -> list[Delivery]:
        """Deliveries after `cursor`, oldest first.

        Strictly greater than, because `seq` is unique. A caller that stores
        the returned `next_cursor` and passes it back sees every delivery
        exactly once.
        """
        statement = (
            select(WebhookDeliveryRow)
            .where(
                WebhookDeliveryRow.endpoint == endpoint,
                WebhookDeliveryRow.seq > cursor,
            )
            .order_by(WebhookDeliveryRow.seq.asc())
            .limit(max(1, min(limit, 500)))
        )
        async with self._database.session() as session:
            rows = (await session.execute(statement)).scalars().all()
        return [self._to_delivery(row) for row in rows]

    async def latest(self, endpoint: str) -> Delivery | None:
        statement = (
            select(WebhookDeliveryRow)
            .where(WebhookDeliveryRow.endpoint == endpoint)
            .order_by(WebhookDeliveryRow.seq.desc())
            .limit(1)
        )
        async with self._database.session() as session:
            row = (await session.execute(statement)).scalars().first()
        return None if row is None else self._to_delivery(row)

    async def stats(self, endpoint: str) -> EndpointStats:
        """Count and freshness for one endpoint.

        The last-received time is what makes a dead sender visible. An
        endpoint that has stopped being called looks exactly like a quiet
        one until you can see how long it has been quiet.
        """
        statement = select(
            func.count(WebhookDeliveryRow.seq),
            func.max(WebhookDeliveryRow.seq),
            func.max(WebhookDeliveryRow.received_at),
        ).where(WebhookDeliveryRow.endpoint == endpoint)
        async with self._database.session() as session:
            count, last_seq, last_at = (await session.execute(statement)).one()
        return EndpointStats(
            count=int(count or 0),
            last_seq=int(last_seq) if last_seq is not None else None,
            last_received_at=_aware(last_at),
        )
