"""What a message can and cannot make this system do."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from index_option_brain.chat.telegram import (
    HELP,
    MAX_MESSAGE,
    BotStats,
    CommandRouter,
    TelegramClient,
    TelegramConfig,
    parse_chat_ids,
    run_bot,
)
from index_option_brain.data.http import HttpResponse
from index_option_brain.database.engine import Database
from index_option_brain.events.calendar_store import CalendarStore, EntryKind

TOKEN = "1234567890:AAExampleTokenValueHere"
NOW = datetime(2026, 9, 10, 6, 0, tzinfo=UTC)
ME = 4242


class FakeSession:
    """Records outbound calls and answers from a scripted table."""

    def __init__(self, answers: dict[str, HttpResponse] | None = None) -> None:
        self.answers = answers or {}
        self.gets: list[tuple[str, dict[str, str]]] = []
        self.posts: list[dict[str, Any]] = []
        self.raise_on_get = False

    async def get(
        self,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.gets.append((url, dict(params or {})))
        if self.raise_on_get:
            raise OSError("connection refused")
        for fragment, response in self.answers.items():
            if fragment in url:
                return response
        return HttpResponse(status_code=404, text="not found")

    async def post(
        self,
        url: str,
        *,
        json: Any | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.posts.append({"url": url, "json": json})
        return HttpResponse(status_code=200, text='{"ok":true}')

    async def delete(self, url: str, **kw: Any) -> HttpResponse:  # pragma: no cover
        raise AssertionError("the bot must never DELETE")

    async def aclose(self) -> None:
        return None


@pytest.fixture
async def database() -> AsyncIterator[Database]:
    db = Database.in_memory()
    await db.create_schema()
    yield db
    await db.aclose()


def config(**over: Any) -> TelegramConfig:
    kwargs: dict[str, Any] = {"token": TOKEN, "allowed_chat_ids": frozenset({ME})}
    kwargs.update(over)
    return TelegramConfig(**kwargs)


def router_over(
    database: Database, session: FakeSession, *, environ: dict[str, str] | None = None
) -> CommandRouter:
    return CommandRouter(
        CalendarStore(database),
        session,
        console_base_url="http://127.0.0.1:8000",
        clock=lambda: NOW,
        environ=environ if environ is not None else {"SIGNAL_RELAY_KILL_FILE": "/nonexistent/k"},
    )


class TestWhoMayTalkToIt:
    def test_an_allowlist_is_required(self) -> None:
        """A bot with no allowlist answers anyone who finds it, and what it
        answers with is what your system is doing."""
        with pytest.raises(ValueError, match="allowed_chat_ids is required"):
            config(allowed_chat_ids=frozenset())

    def test_a_token_that_is_not_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Telegram bot token"):
            config(token="nope")

    def test_a_bad_chat_id_is_refused_not_skipped(self) -> None:
        """Skipping it would silently narrow who can reach the bot, and the
        symptom is the bot ignoring you for no visible reason."""
        with pytest.raises(ValueError, match="numeric chat id"):
            parse_chat_ids("4242,oops")

    def test_ids_parse(self) -> None:
        assert parse_chat_ids(" 1, 2 ,3 ") == frozenset({1, 2, 3})

    async def test_a_stranger_gets_no_reply_at_all(self, database: Database) -> None:
        """Not an error — an error confirms the bot is real and attached to
        something worth probing."""
        session = FakeSession()
        client = TelegramClient(config(), session)
        updates = [
            {"update_id": 1, "message": {"chat": {"id": 999}, "text": "/status"}}
        ]
        session.answers = {"getUpdates": HttpResponse(200, '{"ok":true,"result":' + str(updates).replace("'", '"') + "}")}
        stats = await run_bot(
            config(), client, router_over(database, session), max_iterations=1
        )
        assert stats.ignored == 1
        assert stats.handled == 0
        assert session.posts == []   # nothing was sent back
        assert 999 in stats.refused


class TestReadsOnly:
    async def test_help_lists_the_commands(self, database: Database) -> None:
        answer = await router_over(database, FakeSession()).handle("/help", "me")
        assert answer == HELP
        assert "/kill is one-way" in HELP

    async def test_a_plain_message_is_ignored(self, database: Database) -> None:
        assert await router_over(database, FakeSession()).handle("hello", "me") is None

    async def test_brief_is_relayed_from_the_console(self, database: Database) -> None:
        session = FakeSession(
            {"/api/brief/NIFTY": HttpResponse(200, "NIFTY 24025.65\nNothing is authorized")}
        )
        answer = await router_over(database, session).handle("/brief NIFTY", "me")
        assert answer is not None
        assert "Nothing is authorized" in answer

    async def test_brief_sanitises_the_symbol(self, database: Database) -> None:
        """The symbol goes into a URL path. Anything but alphanumerics is
        dropped rather than escaped, because no real symbol needs them."""
        session = FakeSession({"/api/brief/NIFTY": HttpResponse(200, "ok")})
        await router_over(database, session).handle("/brief ../../etc/passwd", "me")
        assert session.gets[-1][0].endswith("/api/brief/ETCPASSWD")

    async def test_an_unreachable_console_says_so(self, database: Database) -> None:
        """"The console is down" and "the market is quiet" must not look
        alike."""
        session = FakeSession()
        session.raise_on_get = True
        answer = await router_over(database, session).handle("/brief NIFTY", "me")
        assert answer is not None
        assert "unreachable" in answer.lower()

    async def test_alerts_echo_named_fields_only(self, database: Database) -> None:
        """A payload can carry anything a sender put in it, and echoing one
        wholesale into a chat is how a credential reaches a message
        history."""
        body = (
            '{"available":true,"count":1,"last_alert_at":"2026-09-10T05:00:00+00:00",'
            '"alerts":[{"trigger_type":"BREAKOUT","symbol":"NIFTY","price":"24025.65",'
            '"fired_at":"2026-09-10T05:00:00+00:00","secret":"LEAKED-VALUE"}]}'
        )
        session = FakeSession({"/api/tradingview": HttpResponse(200, body)})
        answer = await router_over(database, session).handle("/alerts", "me")
        assert answer is not None
        assert "BREAKOUT" in answer
        assert "LEAKED-VALUE" not in answer

    async def test_an_unknown_command_is_named(self, database: Database) -> None:
        answer = await router_over(database, FakeSession()).handle("/moon", "me")
        assert answer is not None
        assert "/moon" in answer


class TestCalendarDecisions:
    async def test_pending_proposals_say_they_do_not_yet_count(
        self, database: Database
    ) -> None:
        store = CalendarStore(database)
        await store.propose(
            name="RBI MPC",
            starts_at=NOW + timedelta(days=3),
            kind=EntryKind.RBI,
            proposed_by="calendar-agent",
            source="https://rbi.org.in/press",
            now=NOW,
        )
        answer = await router_over(database, FakeSession()).handle("/cal", "me")
        assert answer is not None
        assert "nothing below affects trading yet" in answer
        assert "rbi.org.in" in answer   # provenance is shown, so it can be checked

    async def test_confirming_takes_effect_on_refresh(self, database: Database) -> None:
        store = CalendarStore(database)
        entry = await store.propose(
            name="RBI MPC", starts_at=NOW + timedelta(days=3), now=NOW
        )
        assert entry is not None
        answer = await router_over(database, FakeSession()).handle(
            f"/confirm {entry.seq}", "satish"
        )
        assert answer is not None and "confirmed" in answer
        assert [e.name for e in await store.upcoming(now=NOW)] == ["RBI MPC"]

    async def test_a_decision_is_not_reversible_from_chat(
        self, database: Database
    ) -> None:
        """A blackout appearing and disappearing from a group chat is a
        decision nobody can audit afterwards."""
        store = CalendarStore(database)
        entry = await store.propose(name="Budget", starts_at=NOW + timedelta(days=5), now=NOW)
        assert entry is not None
        router = router_over(database, FakeSession())
        await router.handle(f"/confirm {entry.seq}", "me")
        answer = await router.handle(f"/reject {entry.seq}", "me")
        assert answer is not None
        assert "already CONFIRMED" in answer
        assert len(await store.upcoming(now=NOW)) == 1

    async def test_a_missing_id_says_so(self, database: Database) -> None:
        answer = await router_over(database, FakeSession()).handle("/confirm 99", "me")
        assert answer is not None and "No calendar entry #99" in answer

    async def test_a_non_numeric_id_gets_usage(self, database: Database) -> None:
        answer = await router_over(database, FakeSession()).handle("/confirm soon", "me")
        assert answer is not None and "Usage" in answer


class TestTheKillSwitch:
    async def test_it_engages_and_says_it_is_one_way(
        self, database: Database, tmp_path: Path
    ) -> None:
        marker = tmp_path / "RELAY_KILLED"
        router = router_over(
            database, FakeSession(), environ={"SIGNAL_RELAY_KILL_FILE": str(marker)}
        )
        answer = await router.handle("/kill", "satish")
        assert answer is not None
        assert "engaged" in answer.lower()
        assert "no /unkill" in answer
        assert marker.exists()
        assert "satish" in marker.read_text()

    async def test_there_is_no_command_to_re_arm(self, database: Database) -> None:
        """The asymmetry is the safety property: chat can only ever make
        this system do less."""
        router = router_over(database, FakeSession())
        for attempt in ("/unkill", "/resume", "/enable", "/start_trading"):
            answer = await router.handle(attempt, "me")
            assert answer is None or "Unknown command" in answer or answer == HELP

    async def test_a_second_kill_is_idempotent(
        self, database: Database, tmp_path: Path
    ) -> None:
        marker = tmp_path / "RELAY_KILLED"
        router = router_over(
            database, FakeSession(), environ={"SIGNAL_RELAY_KILL_FILE": str(marker)}
        )
        await router.handle("/kill", "me")
        answer = await router.handle("/kill", "me")
        assert answer is not None and "Already engaged" in answer

    async def test_a_failure_to_engage_is_reported_loudly(
        self, database: Database
    ) -> None:
        """A kill switch that silently failed to engage is worse than not
        having one, because you would stop watching."""
        router = router_over(
            database,
            FakeSession(),
            environ={"SIGNAL_RELAY_KILL_FILE": "/proc/cannot/write/here"},
        )
        answer = await router.handle("/kill", "me")
        assert answer is not None
        assert "COULD NOT ENGAGE" in answer

    async def test_status_reports_the_switch(
        self, database: Database, tmp_path: Path
    ) -> None:
        marker = tmp_path / "RELAY_KILLED"
        marker.write_text("")
        router = router_over(
            database, FakeSession(), environ={"SIGNAL_RELAY_KILL_FILE": str(marker)}
        )
        answer = await router.handle("/killstatus", "me")
        assert answer is not None and "ENGAGED" in answer


class TestTheBotCannotTrade:
    async def test_no_command_places_sizes_or_modifies_an_order(
        self, database: Database
    ) -> None:
        """Asserted as a property of the surface, not of one handler. A
        command that could trade would undo every guard the relay has."""
        router = router_over(database, FakeSession())
        for attempt in (
            "/buy NIFTY 2",
            "/sell 1",
            "/order NIFTY 24000 CE 1",
            "/enable_route ea",
            "/size 5",
            "/exit",
        ):
            answer = await router.handle(attempt, "me")
            assert answer is not None
            assert "Unknown command" in answer

    def test_the_help_text_offers_no_trading_command(self) -> None:
        lowered = HELP.lower()
        for word in ("buy", "sell", "order", "size", "enable"):
            assert word not in lowered

    async def test_it_only_ever_gets_from_the_console(
        self, database: Database
    ) -> None:
        """Reads go through an API that is provably read-only, so a message
        cannot make the system do anything whatever it says."""
        session = FakeSession({"/api": HttpResponse(200, "{}")})
        router = router_over(database, session)
        for command in ("/status", "/brief NIFTY", "/alerts"):
            await router.handle(command, "me")
        assert session.posts == []
        assert all("127.0.0.1:8000" in url for url, _ in session.gets)


class TestTheLoop:
    async def test_a_long_message_is_truncated_not_dropped(
        self, database: Database
    ) -> None:
        """Telegram rejects an over-length message outright, so the failure
        would be silence."""
        session = FakeSession()
        await TelegramClient(config(), session).send(ME, "x" * (MAX_MESSAGE + 500))
        sent = session.posts[0]["json"]["text"]
        assert len(sent) <= MAX_MESSAGE
        assert sent.endswith("truncated")

    async def test_no_parse_mode_is_requested(self, database: Database) -> None:
        """Market data is full of characters Markdown treats as syntax, and
        a message that fails to parse is one Telegram silently drops."""
        session = FakeSession()
        await TelegramClient(config(), session).send(ME, "NIFTY_24000-CE *test*")
        assert "parse_mode" not in session.posts[0]["json"]

    async def test_the_offset_advances_past_refused_updates(
        self, database: Database
    ) -> None:
        """Advancing only past handled updates would make one message from
        a stranger re-fetched forever."""
        session = FakeSession(
            {
                "getUpdates": HttpResponse(
                    200,
                    '{"ok":true,"result":[{"update_id":7,"message":'
                    '{"chat":{"id":999},"text":"/status"}}]}',
                )
            }
        )
        await run_bot(
            config(),
            TelegramClient(config(), session),
            router_over(database, session),
            max_iterations=2,
        )
        offsets = [params.get("offset") for url, params in session.gets if "getUpdates" in url]
        assert offsets == ["0", "8"]

    async def test_a_failing_poll_backs_off_rather_than_spinning(
        self, database: Database
    ) -> None:
        session = FakeSession()
        session.raise_on_get = True
        stats = BotStats()
        await run_bot(
            config(),
            TelegramClient(config(), session),
            router_over(database, session),
            stats=stats,
            max_iterations=2,
        )
        assert stats.errors == 2

    async def test_a_handler_that_raises_does_not_kill_the_loop(
        self, database: Database
    ) -> None:
        class Exploding(CommandRouter):
            async def handle(self, text: str, who: str) -> str | None:
                raise RuntimeError("boom")

        session = FakeSession(
            {
                "getUpdates": HttpResponse(
                    200,
                    '{"ok":true,"result":[{"update_id":1,"message":'
                    '{"chat":{"id":4242},"text":"/status"}}]}',
                )
            }
        )
        router = Exploding(
            CalendarStore(database), session, console_base_url="http://127.0.0.1:8000"
        )
        stats = await run_bot(
            config(), TelegramClient(config(), session), router, max_iterations=1
        )
        assert stats.errors == 1
        assert "failed on my side" in session.posts[-1]["json"]["text"]
