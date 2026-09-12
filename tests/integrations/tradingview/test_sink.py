"""Getting an accepted alert across the process boundary."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from index_option_brain.contracts.enums import TriggerType
from index_option_brain.database.engine import Database
from index_option_brain.database.models import SystemEventRow
from index_option_brain.integrations.tradingview.alert import alert_from_payload
from index_option_brain.integrations.tradingview.auth import WebhookGuard, WebhookGuardConfig
from index_option_brain.integrations.tradingview.inbox import AlertInbox
from index_option_brain.integrations.tradingview.receiver import create_webhook_app
from index_option_brain.integrations.tradingview.sink import (
    ALERT_KIND,
    DatabaseAlertSink,
    pending_alerts,
)

SECRET = "a-sufficiently-long-secret"
NOW = datetime(2026, 9, 6, 4, 15, 30, tzinfo=UTC)


def alert(**overrides: object):  # type: ignore[no-untyped-def]
    payload: dict[str, object] = {
        "kind": "BREAKOUT",
        "ticker": "NIFTY",
        "interval": "5",
        "price": 24025.65,
        "bar_time": "2026-09-06T04:10:00Z",
        "fired_at": "2026-09-06T04:15:00Z",
    }
    payload.update(overrides)
    return alert_from_payload(payload)


@pytest.fixture
async def database() -> AsyncIterator[Database]:
    db = Database.in_memory()
    await db.create_schema()
    yield db
    await db.aclose()


class TestDatabaseSink:
    async def test_an_alert_becomes_a_readable_event(self, database: Database) -> None:
        await DatabaseAlertSink(database).record(alert().to_event())
        events = await pending_alerts(database)
        assert len(events) == 1
        assert events[0].trigger_type is TriggerType.BREAKOUT
        assert events[0].payload["symbol"] == "NIFTY"

    async def test_the_same_alert_written_twice_reads_back_once(
        self, database: Database
    ) -> None:
        """Two receivers, or one that restarted mid-redelivery, both write
        it. The deterministic event id is what makes that recoverable."""
        event = alert().to_event()
        sink = DatabaseAlertSink(database)
        await sink.record(event)
        await sink.record(event)
        assert len(await pending_alerts(database)) == 1

    async def test_since_drops_what_the_consumer_has_already_read(
        self, database: Database
    ) -> None:
        sink = DatabaseAlertSink(database)
        await sink.record(alert(fired_at="2026-09-06T04:10:00Z").to_event())
        await sink.record(alert(fired_at="2026-09-06T04:20:00Z").to_event())
        recent = await pending_alerts(
            database, since=datetime(2026, 9, 6, 4, 15, tzinfo=UTC)
        )
        assert [e.timestamp for e in recent] == [
            datetime(2026, 9, 6, 4, 20, tzinfo=UTC)
        ]

    async def test_an_alert_sharing_a_second_with_the_cursor_is_not_lost(
        self, database: Database
    ) -> None:
        """Firing time has second resolution and two alerts on the same
        bar close routinely share it. With a strict `>`, a consumer
        storing the last occurred_at it saw would skip the sibling
        permanently — in the table, past the cursor, never returned."""
        sink = DatabaseAlertSink(database)
        await sink.record(alert(kind="BREAKOUT").to_event())
        await sink.record(alert(kind="VWAP_CROSSING").to_event())
        recent = await pending_alerts(
            database, since=datetime(2026, 9, 6, 4, 15, tzinfo=UTC)
        )
        assert len(recent) == 2

    async def test_a_row_of_another_kind_is_not_read_as_an_alert(
        self, database: Database
    ) -> None:
        """`system_events` is a shared operational log. A feed outage is
        not a chart alert and must not enter the pipeline as one."""
        async with database.session() as session:
            session.add(
                SystemEventRow(
                    kind="feed_outage",
                    severity="error",
                    message="NSE unreachable",
                    detail={"event_id": "x", "trigger_type": "BREAKOUT"},
                    occurred_at=NOW,
                )
            )
        assert await pending_alerts(database) == []

    async def test_a_malformed_row_is_skipped_not_repaired(
        self, database: Database
    ) -> None:
        """Guessing at a broken row would put an invented trigger into the
        pipeline."""
        async with database.session() as session:
            session.add(
                SystemEventRow(
                    kind=ALERT_KIND,
                    severity="info",
                    message="mangled",
                    detail={"trigger_type": "NOT_A_TRIGGER"},
                    occurred_at=NOW,
                )
            )
        assert await pending_alerts(database) == []

    async def test_no_secret_is_ever_written(self, database: Database) -> None:
        await DatabaseAlertSink(database).record(alert().to_event())
        async with database.session() as session:
            rows = (await session.execute(SystemEventRow.__table__.select())).all()
        assert SECRET not in json.dumps([dict(row._mapping["detail"]) for row in rows])


class TestReceiverWithASink:
    def test_a_failed_write_answers_503_rather_than_pretending(self) -> None:
        """The alert reached the in-memory inbox, but in a two-process
        deployment that inbox is a dead end. A 200 would tell TradingView's
        log everything worked while nothing downstream will see it."""

        class BrokenSink:
            async def record(self, event: object) -> None:
                raise RuntimeError("disk full")

        guard = WebhookGuard(
            WebhookGuardConfig(secret=SECRET, allowed_ips=frozenset()), clock=lambda: NOW
        )
        client = TestClient(create_webhook_app(guard, AlertInbox(clock=lambda: NOW), BrokenSink()))
        response = client.post(
            "/tv/webhook",
            content=json.dumps(
                {
                    "secret": SECRET,
                    "kind": "BREAKOUT",
                    "ticker": "NIFTY",
                    "interval": "5",
                    "price": 24025.65,
                    "bar_time": "2026-09-06T04:10:00Z",
                    "fired_at": "2026-09-06T04:15:00Z",
                }
            ),
        )
        assert response.status_code == 503
        assert response.json()["reason"] == "NOT_PERSISTED"

    async def test_a_working_sink_persists_what_the_endpoint_accepted(
        self, database: Database
    ) -> None:
        guard = WebhookGuard(
            WebhookGuardConfig(secret=SECRET, allowed_ips=frozenset()), clock=lambda: NOW
        )
        client = TestClient(
            create_webhook_app(
                guard, AlertInbox(clock=lambda: NOW), DatabaseAlertSink(database)
            )
        )
        response = client.post(
            "/tv/webhook",
            content=json.dumps(
                {
                    "secret": SECRET,
                    "kind": "VWAP_CROSSING",
                    "ticker": "NSE:BANKNIFTY",
                    "interval": "5",
                    "price": 52100.4,
                    "bar_time": "2026-09-06T04:10:00Z",
                    "fired_at": "2026-09-06T04:15:00Z",
                }
            ),
        )
        assert response.status_code == 200
        events = await pending_alerts(database)
        assert [e.payload["symbol"] for e in events] == ["BANKNIFTY"]
