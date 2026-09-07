"""The public endpoint: what it accepts, what it says, and what it cannot do."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from index_option_brain.integrations.tradingview.auth import WebhookGuard, WebhookGuardConfig
from index_option_brain.integrations.tradingview.inbox import AlertInbox
from index_option_brain.integrations.tradingview.receiver import create_webhook_app
from tests.integrations.boundary import reachable_offences

SECRET = "a-sufficiently-long-secret"
NOW = datetime(2026, 9, 6, 4, 15, 30, tzinfo=UTC)


def body(**overrides: object) -> str:
    payload: dict[str, object] = {
        "secret": SECRET,
        "kind": "BREAKOUT",
        "ticker": "NIFTY",
        "interval": "5",
        "price": 24025.65,
        "bar_time": "2026-09-06T04:10:00Z",
        "fired_at": "2026-09-06T04:15:00Z",
    }
    payload.update(overrides)
    return json.dumps(payload)


@pytest.fixture
def inbox() -> AlertInbox:
    return AlertInbox(clock=lambda: NOW)


@pytest.fixture
def client(inbox: AlertInbox) -> TestClient:
    # The test client's peer address is "testclient", so the allowlist is
    # emptied here rather than faked — the address check has its own tests.
    guard = WebhookGuard(
        WebhookGuardConfig(secret=SECRET, allowed_ips=frozenset()), clock=lambda: NOW
    )
    return TestClient(create_webhook_app(guard, inbox))


class TestAccepting:
    def test_a_valid_alert_is_queued(self, client: TestClient, inbox: AlertInbox) -> None:
        response = client.post("/tv/webhook", content=body())
        assert response.status_code == 200
        assert response.json()["accepted"] is True
        assert len(inbox) == 1

    def test_a_text_plain_content_type_is_still_parsed(self, client: TestClient) -> None:
        """TradingView sends the alert message verbatim and does not
        reliably set application/json, so the content type is not
        something to route on."""
        response = client.post(
            "/tv/webhook", content=body(), headers={"Content-Type": "text/plain"}
        )
        assert response.status_code == 200

    def test_a_redelivered_bar_answers_200_and_is_not_queued_twice(
        self, client: TestClient, inbox: AlertInbox
    ) -> None:
        """A benign redelivery answered with 4xx puts a red error in
        TradingView's alert log for a webhook that worked."""
        client.post("/tv/webhook", content=body())
        response = client.post("/tv/webhook", content=body())
        assert response.status_code == 200
        assert response.json() == {"ok": True, "accepted": False, "reason": "DUPLICATE"}
        assert len(inbox) == 1


class TestRejecting:
    def test_a_bad_secret_gets_401_and_no_reason(self, client: TestClient) -> None:
        """Naming the reason would tell someone probing the endpoint when
        they have guessed the credential."""
        response = client.post("/tv/webhook", content=body(secret="wrong-but-long-enough"))
        assert response.status_code == 401
        assert response.json() == {"ok": False}

    def test_an_unobservable_trigger_is_refused_with_its_reason(
        self, client: TestClient
    ) -> None:
        """Past the secret, a precise diagnostic is what the operator
        reads out of TradingView's alert log."""
        response = client.post("/tv/webhook", content=body(kind="IV_EXPANSION_COLLAPSE"))
        assert response.status_code == 422
        assert response.json()["reason"] == "NOT_CHART_OBSERVABLE"

    def test_an_unknown_symbol_is_refused(self, client: TestClient) -> None:
        response = client.post("/tv/webhook", content=body(ticker="NSE:FINNIFTY"))
        assert response.status_code == 422
        assert response.json()["reason"] == "UNKNOWN_SYMBOL"

    def test_a_stale_alert_is_refused(self, client: TestClient) -> None:
        response = client.post("/tv/webhook", content=body(fired_at="2026-09-06T03:00:00Z"))
        assert response.status_code == 409
        assert response.json()["reason"] == "STALE"

    def test_an_oversized_body_is_refused(self, client: TestClient) -> None:
        response = client.post("/tv/webhook", content="x" * 9000)
        assert response.status_code == 413

    def test_no_response_ever_echoes_the_body(self, client: TestClient) -> None:
        """Echoing a rejected body back would put the shared secret into
        the HTTP response and from there into TradingView's alert log."""
        for content in [body(secret="wrong-but-long-enough"), body(kind="NONSENSE"), "not json"]:
            response = client.post("/tv/webhook", content=content)
            assert SECRET not in response.text
            assert "wrong-but-long-enough" not in response.text


class TestSurface:
    def test_the_only_write_route_is_the_webhook(self) -> None:
        guard = WebhookGuard(WebhookGuardConfig(secret=SECRET), clock=lambda: NOW)
        app = create_webhook_app(guard, AlertInbox())
        writes = {
            (route.path, method)
            for route in app.routes
            for method in getattr(route, "methods", set())
            if method not in {"GET", "HEAD", "OPTIONS"}
        }
        assert writes == {("/tv/webhook", "POST")}

    def test_health_reports_nothing_about_the_market(self, client: TestClient) -> None:
        """This answers to the public internet. Queue depth and rejection
        counts are information about what the system is watching."""
        response = client.get("/tv/health")
        assert response.json() == {"status": "ok"}

    def test_the_openapi_schema_is_not_published(self, client: TestClient) -> None:
        assert client.get("/openapi.json").status_code == 404
        assert client.get("/docs").status_code == 404


class TestItCannotTrade:
    """The structural claim: an alert cannot place an order because there
    is nothing in this package to place one with."""

    def test_no_module_here_reaches_execution_risk_or_a_broker(self) -> None:
        offences = reachable_offences("index_option_brain.integrations.tradingview")
        assert not offences, f"the receiver can reach order placement: {offences}"

    def test_the_receiver_holds_only_a_guard_an_inbox_and_a_sink(self) -> None:
        """Nothing it is handed could place an order, so no argument can
        smuggle one in."""
        import inspect

        signature = inspect.signature(create_webhook_app)
        assert [p.annotation for p in signature.parameters.values()] == [
            "WebhookGuard",
            "AlertInbox",
            "AlertSink | None",
        ]


class TestTheConsoleStaysReadOnly:
    """Adding an inbound integration must not put a verb on the console.

    The console's read-only proof is what Clawdbot sits behind, and it is a
    property of the whole route table rather than of any one route — so it
    is asserted again here, from the integration that would have been the
    obvious place to break it.
    """

    def test_the_console_has_no_write_route(self) -> None:
        from index_option_brain.app.live import LiveEngine
        from index_option_brain.app.main import create_app

        app = create_app(LiveEngine(), run_poller=False)
        methods = {
            method
            for route in app.routes
            for method in getattr(route, "methods", set())
        }
        assert methods <= {"GET", "HEAD", "OPTIONS"}

    def test_the_console_reads_alerts_and_the_receiver_does_not(self) -> None:
        """The read half is on the tailnet; the public endpoint reports
        nothing about what the system is watching."""
        from index_option_brain.app.live import LiveEngine
        from index_option_brain.app.main import create_app

        console_paths = {getattr(route, "path", "") for route in create_app(
            LiveEngine(), run_poller=False
        ).routes}
        assert "/api/tradingview" in console_paths

        guard = WebhookGuard(WebhookGuardConfig(secret=SECRET), clock=lambda: NOW)
        receiver_paths = {
            getattr(route, "path", "") for route in create_webhook_app(guard, AlertInbox()).routes
        }
        assert not any(path.startswith("/api") for path in receiver_paths)
