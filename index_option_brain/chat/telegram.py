"""Talking to the system from a phone.

Long polling, not a webhook
---------------------------
Telegram offers both, and long polling is the right one here for a
concrete reason: it needs no public URL, no certificate and no inbound
port. It works from behind NAT, on a box whose DNS is not set up yet, and
through a network you do not control. The webhook gateway in this repo is
still waiting on a DNS record; this works today.

It is also the only choice that keeps this process off the public
internet. A webhook receiver for the bot would be a second public
endpoint, and the one this repo already has took three real bugs to get
right.

What it can and cannot do
-------------------------
Reads go through the **console's read-only API** over loopback rather than
by importing the engine. That is deliberate: the console API is provably
read-only — every route GET, HEAD or OPTIONS, with a test asserting it —
so a bot that only GETs it cannot make the system do anything, whatever a
message says.

Three commands write, and each is bounded:

* `/confirm` and `/reject` decide one calendar proposal. A proposal
  changes nothing until decided, and a decision cannot be reversed from
  chat — a blackout appearing and disappearing from a group chat is a
  decision nobody can audit.
* `/kill` engages the relay's kill switch and **there is no /unkill.**
  Re-arming requires deleting a file on the machine. That asymmetry is
  what makes a chat-triggered safety switch safe: chat can only ever move
  the system toward doing less.

There is no command to place, size or modify a trade, and adding one would
undo every guard the relay has.

Who is allowed to talk to it
----------------------------
A bot token is a bearer credential: anyone holding it can message the bot,
and anyone who finds the bot can message it. So `allowed_chat_ids` is
required, and an unknown chat gets **no reply at all** — not an error,
because an error confirms the bot is real and attached to something worth
probing.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from index_option_brain.data.http import HttpError, HttpSession
from index_option_brain.events.calendar_store import (
    CalendarStore,
    EntryState,
)
from index_option_brain.signals.relay import kill_switch_engaged, kill_switch_path

logger = logging.getLogger(__name__)

API_ROOT = "https://api.telegram.org"

#: Telegram's own cap. A longer message is rejected outright, so it is
#: truncated here with a visible marker rather than lost.
MAX_MESSAGE = 4096

#: Seconds Telegram holds an empty long-poll open. Long on purpose: a
#: short timeout is a busy loop against someone else's API.
POLL_TIMEOUT = 25


@dataclass(frozen=True)
class TelegramConfig:
    token: str
    #: Numeric chat ids allowed to talk to the bot. Required — see the
    #: module docstring.
    allowed_chat_ids: frozenset[int]
    console_base_url: str = "http://127.0.0.1:8000"
    poll_timeout: int = POLL_TIMEOUT

    def __post_init__(self) -> None:
        if ":" not in self.token or len(self.token) < 20:
            raise ValueError("that does not look like a Telegram bot token")
        if not self.allowed_chat_ids:
            raise ValueError(
                "allowed_chat_ids is required: without it anyone who finds "
                "the bot can query your trading system"
            )


@dataclass
class BotStats:
    handled: int = 0
    ignored: int = 0
    errors: int = 0
    #: Chat ids that were refused, so a probe is visible rather than only
    #: silent. Bounded — an attacker must not be able to grow this.
    refused: set[int] = field(default_factory=set)


class TelegramClient:
    """The two API calls this bot needs, over the injected HTTP session."""

    def __init__(self, config: TelegramConfig, session: HttpSession) -> None:
        self._config = config
        self._session = session

    def _url(self, method: str) -> str:
        return f"{API_ROOT}/bot{self._config.token}/{method}"

    async def get_updates(self, offset: int) -> list[dict[str, Any]]:
        response = await self._session.get(
            self._url("getUpdates"),
            params={
                "offset": str(offset),
                "timeout": str(self._config.poll_timeout),
                # Only messages. Asking for everything means parsing
                # edited messages, callbacks and channel posts that this
                # bot has no handler for.
                "allowed_updates": '["message"]',
            },
        )
        if not response.is_ok:
            raise HttpError(f"getUpdates answered {response.status_code}")
        body = response.json()
        if not isinstance(body, dict) or not body.get("ok"):
            raise HttpError(f"getUpdates was not ok: {str(body)[:200]}")
        result = body.get("result") or []
        return [item for item in result if isinstance(item, dict)]

    async def send(self, chat_id: int, text: str) -> None:
        trimmed = text if len(text) <= MAX_MESSAGE else text[: MAX_MESSAGE - 20] + "\n… truncated"
        response = await self._session.post(
            self._url("sendMessage"),
            json={
                "chat_id": chat_id,
                "text": trimmed,
                # No parse mode. Market data is full of characters Markdown
                # and HTML both treat as syntax — an underscore in a
                # strategy name, a hyphen in a symbol — and a message that
                # fails to parse is one Telegram silently drops.
                "disable_web_page_preview": True,
            },
        )
        if not response.is_ok:
            raise HttpError(f"sendMessage answered {response.status_code}")


HELP = """Index Brain

Reads
  /status          engine, poller and calendar state
  /brief NIFTY     what the engine sees and decided
  /alerts          chart alerts accepted, and how stale
  /cal             calendar: pending proposals and confirmed events

Decisions
  /confirm <id>    accept a calendar proposal
  /reject <id>     discard one

Safety
  /kill            stop every relay route now
  /killstatus      whether it is engaged

/kill is one-way. Re-arming needs the machine — chat can only ever make
this system do less."""


class CommandRouter:
    """Maps a message to an answer, and holds nothing that can trade."""

    def __init__(
        self,
        store: CalendarStore,
        session: HttpSession,
        *,
        console_base_url: str,
        clock: Callable[[], datetime] | None = None,
        environ: dict[str, str] | None = None,
    ) -> None:
        self._store = store
        self._session = session
        self._console = console_base_url.rstrip("/")
        self._clock = clock or (lambda: datetime.now(UTC))
        self._environ = environ

    async def _console_text(self, path: str) -> str:
        try:
            response = await self._session.get(f"{self._console}{path}")
        except (HttpError, OSError) as exc:
            # Said plainly rather than as an empty answer. "The console is
            # unreachable" and "the market is quiet" must not look alike.
            return f"Console unreachable at {self._console} — {exc}"
        if not response.is_ok:
            return f"Console answered {response.status_code} for {path}"
        return response.text

    async def _console_json(self, path: str) -> Any:
        try:
            response = await self._session.get(f"{self._console}{path}")
        except (HttpError, OSError) as exc:
            return {"unreachable": str(exc)}
        if not response.is_ok:
            return {"unreachable": f"HTTP {response.status_code}"}
        try:
            return response.json()
        except ValueError:
            return {"unreachable": "the console did not answer JSON"}

    async def status(self) -> str:
        body = await self._console_json("/api/status")
        if isinstance(body, dict) and "unreachable" in body:
            return f"Console unreachable — {body['unreachable']}"
        lines = ["Engine"]
        if isinstance(body, dict):
            for key in ("session", "run_mode", "providers_ready", "poller"):
                if key in body:
                    lines.append(f"  {key}: {body[key]}")
        pending = await self._store.pending(limit=1)
        upcoming = await self._store.upcoming(limit=1, now=self._clock())
        lines.append(
            f"Calendar: {len(pending)} pending, "
            f"{'next ' + upcoming[0].name if upcoming else 'nothing confirmed ahead'}"
        )
        lines.append(f"Relay kill switch: {'ENGAGED' if self._killed() else 'off'}")
        return "\n".join(lines)

    async def brief(self, symbol: str) -> str:
        clean = "".join(c for c in symbol.upper() if c.isalnum())[:16]
        if not clean:
            return "Usage: /brief NIFTY"
        return await self._console_text(f"/api/brief/{clean}")

    async def alerts(self) -> str:
        body = await self._console_json("/api/tradingview?limit=5")
        if not isinstance(body, dict):
            return "The console did not answer as expected."
        if "unreachable" in body:
            return f"Console unreachable — {body['unreachable']}"
        if not body.get("available"):
            return f"Alerts unavailable: {body.get('reason', 'no reason given')}"
        last = body.get("last_alert_at")
        lines = [f"{body.get('count', 0)} alert(s) recorded", f"last: {last or 'never'}"]
        for alert in (body.get("alerts") or [])[:5]:
            # Named fields only. A payload can carry anything a sender put
            # in it, and echoing one wholesale into a chat is how a
            # credential ends up in a message history.
            lines.append(
                f"  {alert.get('trigger_type')} {alert.get('symbol')} "
                f"@ {alert.get('price')} ({alert.get('fired_at')})"
            )
        return "\n".join(lines)

    async def calendar(self) -> str:
        pending = await self._store.pending()
        upcoming = await self._store.upcoming(now=self._clock())
        lines: list[str] = []
        if pending:
            lines.append(f"Pending — nothing below affects trading yet ({len(pending)}):")
            lines.extend(entry.describe() for entry in pending)
            lines.append("\n/confirm <id> or /reject <id>")
        else:
            lines.append("No pending proposals.")
        lines.append("")
        if upcoming:
            lines.append(f"Confirmed and ahead ({len(upcoming)}):")
            lines.extend(
                f"#{e.seq}  {e.starts_at.strftime('%a %d %b %H:%M UTC')}  {e.name}"
                + ("" if e.blocks_new_entries else "  (does not block entries)")
                for e in upcoming
            )
        else:
            lines.append("No confirmed events ahead. No event blackout is in effect.")
        return "\n".join(lines)

    async def decide(self, argument: str, state: EntryState, who: str) -> str:
        if not argument.strip().lstrip("#").isdigit():
            return f"Usage: /{state.lower()} <id> — see /cal"
        seq = int(argument.strip().lstrip("#"))
        entry = await self._store.decide(
            seq, state=state, decided_by=who, now=self._clock()
        )
        if entry is None:
            return f"No calendar entry #{seq}."
        if entry.state is not state:
            return f"#{seq} was already {entry.state} — decisions are not reversible from chat."
        verb = "confirmed" if state is EntryState.CONFIRMED else "rejected"
        note = (
            "\nIt will take effect on the calendar's next refresh."
            if state is EntryState.CONFIRMED
            else ""
        )
        return f"#{seq} {verb}: {entry.name} at {entry.starts_at.isoformat()}{note}"

    def _killed(self) -> bool:
        return kill_switch_engaged(self._environ)

    def kill(self, who: str) -> str:
        path = kill_switch_path(self._environ)
        if self._killed():
            return f"Already engaged. No route will send.\nFile: {path}"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                f"engaged from chat by {who} at {self._clock().isoformat()}\n"
            )
        except OSError as exc:
            # Reported, never swallowed. A kill switch that silently failed
            # to engage is worse than not having one, because you would
            # stop watching.
            return f"COULD NOT ENGAGE: {exc}\nStop the gateway process instead."
        return (
            "Kill switch engaged. No relay route will send.\n"
            f"File: {path}\n"
            "There is no /unkill — delete that file on the machine to re-arm."
        )

    def kill_status(self) -> str:
        path = kill_switch_path(self._environ)
        if self._killed():
            return f"ENGAGED — no route will send.\nFile: {path}"
        return f"Off — enabled routes will send.\nWould create: {path}"

    async def handle(self, text: str, who: str) -> str | None:
        """The answer to one message, or None to stay silent."""
        stripped = text.strip()
        if not stripped.startswith("/"):
            return None  # not addressed to the bot in any useful way
        # "/cal@MyBot arg" in a group.
        head, _, rest = stripped.partition(" ")
        command = head.split("@", 1)[0].lower()

        if command in ("/start", "/help"):
            return HELP
        if command == "/status":
            return await self.status()
        if command == "/brief":
            return await self.brief(rest or "NIFTY")
        if command == "/alerts":
            return await self.alerts()
        if command == "/cal":
            return await self.calendar()
        if command == "/confirm":
            return await self.decide(rest, EntryState.CONFIRMED, who)
        if command == "/reject":
            return await self.decide(rest, EntryState.REJECTED, who)
        if command == "/kill":
            return self.kill(who)
        if command == "/killstatus":
            return self.kill_status()
        return f"Unknown command {command}. /help lists them."


async def run_bot(
    config: TelegramConfig,
    client: TelegramClient,
    router: CommandRouter,
    *,
    stats: BotStats | None = None,
    max_iterations: int | None = None,
) -> BotStats:
    """The long-poll loop.

    `max_iterations` exists so a test can run it a fixed number of times;
    production leaves it None.

    The offset is advanced past every update *examined*, including the ones
    from chats that are not allowed. Advancing only past handled updates
    would make one message from a stranger re-fetched forever.
    """
    counters = stats or BotStats()
    offset = 0
    iterations = 0
    while max_iterations is None or iterations < max_iterations:
        iterations += 1
        try:
            updates = await client.get_updates(offset)
        except (HttpError, OSError) as exc:
            counters.errors += 1
            logger.warning("getUpdates failed: %s", exc)
            await asyncio.sleep(min(30, 2 * counters.errors))
            continue

        for update in updates:
            offset = max(offset, int(update.get("update_id", 0)) + 1)
            message = update.get("message") or {}
            chat = message.get("chat") or {}
            chat_id = chat.get("id")
            text = message.get("text")
            if not isinstance(chat_id, int) or not isinstance(text, str):
                continue
            if chat_id not in config.allowed_chat_ids:
                counters.ignored += 1
                if len(counters.refused) < 100:
                    counters.refused.add(chat_id)
                # No reply. An error would confirm the bot is real and
                # attached to something worth probing.
                logger.warning("ignored a message from chat %s", chat_id)
                continue
            who = str(
                (message.get("from") or {}).get("username")
                or (message.get("from") or {}).get("id")
                or chat_id
            )
            try:
                answer = await router.handle(text, who)
            except Exception:
                counters.errors += 1
                logger.exception("handler failed for %r", text.split(" ")[0])
                answer = "That failed on my side. The log has the detail."
            if answer is None:
                continue
            try:
                await client.send(chat_id, answer)
                counters.handled += 1
            except (HttpError, OSError) as exc:
                counters.errors += 1
                logger.warning("sendMessage failed: %s", exc)
    return counters


def parse_chat_ids(raw: str) -> frozenset[int]:
    """Parse `TELEGRAM_ALLOWED_CHAT_IDS`, refusing anything unparseable.

    Refused rather than skipped: a typo in one id would otherwise silently
    narrow who can reach the bot, and the symptom is the bot ignoring you
    for no visible reason.
    """
    ids: set[int] = set()
    for part in raw.replace(" ", "").split(","):
        if not part:
            continue
        try:
            ids.add(int(part))
        except ValueError as exc:
            raise ValueError(f"{part!r} is not a numeric chat id") from exc
    return frozenset(ids)


def kill_file_note(environ: dict[str, str] | None = None) -> str:
    return str(kill_switch_path(environ))


__all__ = [
    "HELP",
    "MAX_MESSAGE",
    "BotStats",
    "CommandRouter",
    "TelegramClient",
    "TelegramConfig",
    "kill_file_note",
    "parse_chat_ids",
    "run_bot",
]
