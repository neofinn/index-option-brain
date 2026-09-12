"""What the gateway process refuses to do at startup."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from index_option_brain.config.settings import Settings
from index_option_brain.integrations.tradingview.inbox import AlertInbox
from index_option_brain.integrations.webhooks.__main__ import main, tradingview_handler
from index_option_brain.integrations.webhooks.endpoints import Endpoint, EndpointKind
from index_option_brain.integrations.webhooks.gateway import HandlerNote

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
    """The clock is pinned in every case here.

    The fixture timestamps are fixed, and the handler now enforces a
    freshness window — so without a pinned clock these tests would pass on
    the day they were written and fail from the next one, which is a
    failure mode this suite has already had once.
    """

    NOW = datetime(2026, 9, 7, 4, 15, 30, tzinfo=UTC)

    def _endpoint(self) -> Endpoint:
        return Endpoint(
            slug="tv",
            ingest_secret=INGEST,
            read_token=READ,
            kind=EndpointKind.TRADINGVIEW,
        )

    def _alert(self, **over: object) -> dict[str, object]:
        body: dict[str, object] = {
            "kind": "BREAKOUT",
            "ticker": "NIFTY",
            "interval": "5",
            "price": 24025.65,
            "bar_time": "2026-09-07T04:10:00Z",
            "fired_at": "2026-09-07T04:15:00Z",
        }
        body.update(over)
        return body

    async def test_a_valid_alert_is_accepted(self) -> None:
        recorded: list[object] = []

        class Sink:
            async def record(self, event: object) -> None:
                recorded.append(event)

        handle = tradingview_handler(
            AlertInbox(), Sink(), clock=lambda: self.NOW  # type: ignore[arg-type]
        )
        assert await handle(self._endpoint(), self._alert()) == "accepted"
        assert len(recorded) == 1

    async def test_an_unobservable_claim_is_the_senders_error(self) -> None:
        """A 200 would put a green tick in TradingView's alert log for an
        alert that did nothing, and that log is the only place the
        operator is looking."""

        class Sink:
            async def record(self, event: object) -> None:
                raise AssertionError("should not be reached")

        handle = tradingview_handler(
            AlertInbox(), Sink(), clock=lambda: self.NOW  # type: ignore[arg-type]
        )
        note = await handle(self._endpoint(), self._alert(kind="IV_EXPANSION_COLLAPSE"))
        assert isinstance(note, HandlerNote)
        assert note.note == "NOT_CHART_OBSERVABLE"
        assert note.sender_error is True

    async def test_a_stale_alert_is_refused_on_this_path_too(self) -> None:
        """The gap a live harness run found. WebhookGuard enforces
        freshness on the standalone receiver, and this second way in was
        written without it — so a twenty-minute-old breakout arriving
        through the gateway was accepted and woke the pipeline."""

        class Sink:
            async def record(self, event: object) -> None:
                raise AssertionError("a stale alert must not be recorded")

        handle = tradingview_handler(
            AlertInbox(), Sink(), clock=lambda: self.NOW  # type: ignore[arg-type]
        )
        note = await handle(
            self._endpoint(), self._alert(fired_at="2026-09-07T03:55:00Z")
        )
        assert isinstance(note, HandlerNote)
        assert note.note == "STALE"
        assert note.sender_error is True

    async def test_a_duplicate_is_not_the_senders_error(self) -> None:
        """TradingView re-fires on a reconnect. The correct answer is
        "already handled", which is a success, not something to show red
        in their log."""

        class Sink:
            async def record(self, event: object) -> None:
                return None

        handle = tradingview_handler(
            AlertInbox(), Sink(), clock=lambda: self.NOW  # type: ignore[arg-type]
        )
        endpoint = self._endpoint()
        await handle(endpoint, self._alert())
        note = await handle(endpoint, self._alert(fired_at="2026-09-07T04:15:20Z"))
        assert isinstance(note, HandlerNote)
        assert note.note == "DUPLICATE"
        assert note.sender_error is False
