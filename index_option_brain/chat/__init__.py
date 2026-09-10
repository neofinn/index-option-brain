"""Chat as an operator surface.

Reads go through the console's read-only API, so a message cannot make the
system do anything. Three commands write, and each is bounded: two decide
a calendar proposal, and one engages the relay's kill switch — which has
no counterpart, because chat should only ever be able to make this system
do less.
"""

from index_option_brain.chat.telegram import (
    HELP,
    BotStats,
    CommandRouter,
    TelegramClient,
    TelegramConfig,
    parse_chat_ids,
    run_bot,
)

__all__ = [
    "HELP",
    "BotStats",
    "CommandRouter",
    "TelegramClient",
    "TelegramConfig",
    "parse_chat_ids",
    "run_bot",
]
