"""The pull side an Expert Advisor reads."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from index_option_brain.data.http import HttpResponse
from index_option_brain.database.engine import Database
from index_option_brain.integrations.webhooks.endpoints import Endpoint
from index_option_brain.signals.feed import CSV_COLUMNS, create_signal_feed_router
from index_option_brain.signals.relay import SignalRelay
from index_option_brain.signals.routes import PullDestination, SignalRoute

READ = "read-token-long-enough-too"
INGEST = "ingest-secret-long-enough"
NOW = datetime(2026, 9, 7, 4, 15, 30, tzinfo=UTC)


class NoSession:
    async def post(self, url: str, **kw: Any) -> HttpResponse:  # pragma: no cover
        raise AssertionError("a pull route must not call out")

    async def get(self, url: str, **kw: Any) -> HttpResponse:  # pragma: no cover
        raise AssertionError("unused")

    async def delete(self, url: str, **kw: Any) -> HttpResponse:  # pragma: no cover
        raise AssertionError("unused")

    async def aclose(self) -> None:
        return None


def payload(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "strategy": "orb-nifty",
        "ticker": "NIFTY",
        "action": "buy",
        "quantity": 1,
        "bar_time": "2026-09-07T04:10:00Z",
        "fired_at": "2026-09-07T04:15:00Z",
    }
    body.update(overrides)
    return body


@pytest.fixture
async def database() -> AsyncIterator[Database]:
    db = Database.in_memory()
    await db.create_schema()
    yield db
    await db.aclose()


def build(
    database: Database, *, enabled: bool = True
) -> tuple[TestClient, SignalRelay]:
    routes = {
        "ea": SignalRoute(
            name="ea",
            destination=PullDestination(),
            symbol_map={"NIFTY": "NIFTY.I"},
            enabled=enabled,
            max_quantity=Decimal(2),
            max_orders_per_day=50,
        )
    }
    endpoints = {"ea": Endpoint(slug="ea", ingest_secret=INGEST, read_token=READ)}
    relay = SignalRelay(
        routes, database, session=NoSession(), clock=lambda: NOW, environ={}
    )
    app = FastAPI()
    app.include_router(create_signal_feed_router(relay, endpoints))
    return TestClient(app), relay


HEAD: Mapping[str, str] = {"Authorization": f"Bearer {READ}"}


class TestCsvFeed:
    async def test_the_cursor_comes_first_then_the_header(
        self, database: Database
    ) -> None:
        """So an EA can read the cursor without counting lines."""
        client, relay = build(database)
        await relay.dispatch("ea", payload())
        lines = client.get("/v1/ea/signals.csv", headers=HEAD).text.strip().split("\n")
        assert lines[0].startswith("#cursor=")
        assert lines[1] == ",".join(CSV_COLUMNS)
        assert "NIFTY.I" in lines[2]

    async def test_the_cursor_is_also_a_header(self, database: Database) -> None:
        client, relay = build(database)
        await relay.dispatch("ea", payload())
        response = client.get("/v1/ea/signals.csv", headers=HEAD)
        assert response.headers["X-Next-Cursor"] == "1"

    async def test_free_text_commas_cannot_shift_a_column(
        self, database: Database
    ) -> None:
        """MQL's StringSplit has no notion of quoting, so a quoted field
        would shift every column after it — and an EA reads by index."""
        client, relay = build(database)
        await relay.dispatch("ea", payload(strategy="orb, v2"))
        rows = client.get("/v1/ea/signals.csv", headers=HEAD).text.strip().split("\n")
        assert len(rows[2].split(",")) == len(CSV_COLUMNS)

    async def test_a_dry_run_signal_is_withheld(self, database: Database) -> None:
        """The whole meaning of a dry run: the rehearsal is recorded and
        nothing downstream acts on it."""
        client, relay = build(database, enabled=False)
        await relay.dispatch("ea", payload())
        lines = client.get("/v1/ea/signals.csv", headers=HEAD).text.strip().split("\n")
        assert len(lines) == 2  # cursor + header, no rows

    async def test_a_blocked_signal_is_withheld(self, database: Database) -> None:
        client, relay = build(database)
        await relay.dispatch("ea", payload(quantity=99))
        lines = client.get("/v1/ea/signals.csv", headers=HEAD).text.strip().split("\n")
        assert len(lines) == 2

    async def test_the_cursor_advances_past_withheld_rows(
        self, database: Database
    ) -> None:
        """Advancing only past actionable rows would re-examine a blocked
        signal on every poll for as long as it is the newest row."""
        client, relay = build(database)
        await relay.dispatch("ea", payload(quantity=99))
        response = client.get("/v1/ea/signals.csv", headers=HEAD)
        assert response.headers["X-Next-Cursor"] != "0"

    def test_an_unauthorized_read_is_plain_text_too(self, database: Database) -> None:
        """An EA parsing this cannot handle a JSON error body; a #error
        line it can read positionally is what it gets."""
        client, _ = build(database)
        response = client.get("/v1/ea/signals.csv")
        assert response.status_code == 401
        assert response.text.startswith("#error=")


class TestJsonFeed:
    async def test_it_returns_actionable_signals(self, database: Database) -> None:
        client, relay = build(database)
        await relay.dispatch("ea", payload())
        body = client.get("/v1/ea/signals", headers=HEAD).json()
        assert body["count"] == 1
        assert body["signals"][0]["symbol"] == "NIFTY.I"
        assert body["signals"][0]["action"] == "buy"

    async def test_the_cursor_round_trips(self, database: Database) -> None:
        client, relay = build(database)
        await relay.dispatch("ea", payload())
        first = client.get("/v1/ea/signals", headers=HEAD).json()
        again = client.get(
            f"/v1/ea/signals?since={first['next_cursor']}", headers=HEAD
        ).json()
        assert again["count"] == 0


class TestRouteState:
    async def test_it_reports_whether_the_route_would_actually_send(
        self, database: Database
    ) -> None:
        """A dry run produces no rows, which looks identical to a quiet
        strategy from the feed alone."""
        client, _ = build(database, enabled=False)
        body = client.get("/v1/ea/routes", headers=HEAD).json()
        assert body["enabled"] is False
        assert body["destination"] == "pull"
        assert body["symbols"] == ["NIFTY"]
        assert body["max_quantity"] == "2"

    def test_an_endpoint_with_no_route_says_so(self, database: Database) -> None:
        endpoints = {
            "other": Endpoint(slug="other", ingest_secret=INGEST, read_token=READ)
        }
        relay = SignalRelay({}, database, clock=lambda: NOW, environ={})
        app = FastAPI()
        app.include_router(create_signal_feed_router(relay, endpoints))
        body = TestClient(app).get("/v1/other/routes", headers=HEAD).json()
        assert body["configured"] is False


class TestAuth:
    def test_a_wrong_token_is_refused(self, database: Database) -> None:
        client, _ = build(database)
        assert (
            client.get(
                "/v1/ea/signals", headers={"Authorization": "Bearer wrong-token-here"}
            ).status_code
            == 401
        )

    def test_a_query_token_works_for_an_ea(self, database: Database) -> None:
        """MT4's WebRequest can set headers, but a query token is one less
        thing to get wrong in MQL."""
        client, _ = build(database)
        assert client.get(f"/v1/ea/signals?token={READ}").status_code == 200

    def test_an_unknown_slug_is_a_404(self, database: Database) -> None:
        client, _ = build(database)
        assert client.get("/v1/nope/signals", headers=HEAD).status_code == 404

    def test_the_feed_exposes_no_write_route(self, database: Database) -> None:
        client, _ = build(database)
        methods = {
            method
            for route in client.app.routes  # type: ignore[attr-defined]
            for method in getattr(route, "methods", set())
        }
        assert methods <= {"GET", "HEAD", "OPTIONS"}
