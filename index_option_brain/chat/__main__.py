"""Run the chat bot: `python -m index_option_brain.chat`.

A fourth process, and the only one that needs no inbound reachability at
all — it dials out to Telegram and polls. That makes it the one operator
surface that works before the DNS record exists.

It refuses to start without an allowlist, for the same reason the gateway
refuses to start without a secret: a bot anyone who finds it can query is
worse than no bot.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from index_option_brain.chat.telegram import (
    CommandRouter,
    TelegramClient,
    TelegramConfig,
    parse_chat_ids,
    run_bot,
)
from index_option_brain.config.settings import Settings, get_settings
from index_option_brain.data.http import HttpxSession
from index_option_brain.database.engine import Database, sqlite_url
from index_option_brain.events.calendar_store import CalendarStore
from index_option_brain.signals.relay import kill_switch_path

logger = logging.getLogger(__name__)


def build_config(settings: Settings) -> TelegramConfig:
    return TelegramConfig(
        token=settings.telegram_bot_token,
        allowed_chat_ids=parse_chat_ids(settings.telegram_allowed_chat_ids),
        console_base_url=settings.console_base_url,
    )


async def serve(settings: Settings) -> int:
    config = build_config(settings)
    database = Database(url=settings.database_url or sqlite_url(settings.sqlite_path))
    await database.create_schema()

    session = HttpxSession()
    router = CommandRouter(
        CalendarStore(database),
        session,
        console_base_url=config.console_base_url,
    )
    logger.info(
        "chat bot polling for %d allowed chat(s); console at %s; kill file %s",
        len(config.allowed_chat_ids),
        config.console_base_url,
        kill_switch_path(),
    )
    try:
        await run_bot(config, TelegramClient(config, session), router)
    finally:
        await session.aclose()
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = get_settings()
    if not settings.telegram_bot_token:
        logger.error("TELEGRAM_BOT_TOKEN is not set; nothing to do")
        return 2
    try:
        build_config(settings)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    try:
        return asyncio.run(serve(settings))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
