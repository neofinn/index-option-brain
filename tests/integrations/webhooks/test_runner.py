"""What the gateway process refuses to do at startup."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from index_option_brain.config.settings import Settings
from index_option_brain.integrations.tradingview.inbox import AlertInbox
from index_option_brain.integrations.webhooks.__main__ import main, tradingview_handler
from index_option_brain.integrations.webhooks.endpoints import Endpoint, EndpointKind

INGEST = "ingest-secret-long-enough"
READ = "read-token-long-enough-too"


def _settings(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setattr(
        "index_option_brain.integrations.webhooks.__main__.get_settings",
        lambda: Settings(WEBHOOK_ENDPOINTS_FILE=str(path)),
    )


class TestStartup:
    def test_it_refuses_to_start_with_no_registry(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A gateway with nothing configured is a public port that answers
        404 to everything, which looks like a working deployment."""
        _settings(monkeypatch, tmp_path / "absent.json")
        assert main() == 2

    def test_it_refuses_to_start_on_a_bad_registry(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        file = tmp_path / "endpoints.json"
        file.write_text(json.dumps({"tv": {"ingest_secret": "short", "read_token": READ}}))
        file.chmod(0o600)
        _settings(monkeypatch, file)
        assert main() == 2

    def test_it_refuses_a_registry_anyone_can_read(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        file = tmp_path / "endpoints.json"
        file.write_text(json.dumps({"tv": {"ingest_secret": INGEST, "read_token": READ}}))
        file.chmod(0o644)
        _settings(monkeypatch, file)
        assert main() == 2


class TestTradingViewHandler:
    async def test_a_valid_alert_is_accepted(self) -> None:
        recorded: list[object] = []

        class Sink:
            async def record(self, event: object) -> None:
                recorded.append(event)

        handle = tradingview_handler(AlertInbox(), Sink())  # type: ignore[arg-type]
        endpoint = Endpoint(
            slug="tv",
            ingest_secret=INGEST,
            read_token=READ,
            kind=EndpointKind.TRADINGVIEW,
        )
        note = await handle(
            endpoint,
            {
                "kind": "BREAKOUT",
                "ticker": "NIFTY",
                "interval": "5",
                "price": 24025.65,
                "bar_time": "2026-09-07T04:10:00Z",
                "fired_at": "2026-09-07T04:15:00Z",
            },
        )
        assert note == "accepted"
        assert len(recorded) == 1

    async def test_an_unobservable_claim_is_reported_not_raised(self) -> None:
        """The reason reaches TradingView's own alert log this way. A
        rejection only a server log knows about is found out about days
        later."""
        class Sink:
            async def record(self, event: object) -> None:
                raise AssertionError("should not be reached")

        handle = tradingview_handler(AlertInbox(), Sink())  # type: ignore[arg-type]
        endpoint = Endpoint(
            slug="tv",
            ingest_secret=INGEST,
            read_token=READ,
            kind=EndpointKind.TRADINGVIEW,
        )
        note = await handle(
            endpoint,
            {
                "kind": "IV_EXPANSION_COLLAPSE",
                "ticker": "NIFTY",
                "interval": "5",
                "price": 24025.65,
                "bar_time": "2026-09-07T04:10:00Z",
                "fired_at": "2026-09-07T04:15:00Z",
            },
        )
        assert note == "NOT_CHART_OBSERVABLE"
