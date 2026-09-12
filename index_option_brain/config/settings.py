from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RunMode(StrEnum):
    """Supported execution modes. The same brain must run in every mode (spec §22)."""

    LIVE = "live"
    PAPER = "paper"
    BACKTEST = "backtest"
    REPLAY = "replay"


class Settings(BaseSettings):
    """Process-wide configuration.

    ``llm_enabled`` must default to False: the deterministic quantitative brain
    is the mandatory decision path (spec §23, §35). The LLM/agent layer is an
    optional add-on that the trading engine must never require to exist.
    """

    model_config = SettingsConfigDict(env_file=".env", env_prefix="", case_sensitive=False)

    llm_enabled: bool = Field(default=False, alias="LLM_ENABLED")
    run_mode: RunMode = Field(default=RunMode.PAPER, alias="RUN_MODE")

    database_url: str = Field(default="", alias="DATABASE_URL")
    """Where observations are persisted. Empty means SQLite at `sqlite_path`.

    The default used to be a Postgres URL on localhost, which meant a fresh
    box's first act was to fail to connect to a server nobody had installed
    — and to record nothing while it did. Capture cannot be back-filled, so
    the default has to be a store that always works. A real Postgres URL
    here takes over; both run the same schema.

    Synchronous forms are accepted: `postgresql://` and `sqlite:///` are
    rewritten onto async drivers rather than failing at connect time with an
    error about greenlets.
    """
    sqlite_path: str = Field(default="var/index_brain.sqlite", alias="SQLITE_PATH")
    capture_enabled: bool = Field(default=True, alias="CAPTURE_ENABLED")
    """Whether to record what is observed.

    On by default, and worth defending: the chain corpus is the only thing
    this system accumulates that cannot be bought or recovered later. A
    session not captured is a session of future backtesting that does not
    exist.
    """
    capture_chain_seconds: int = Field(default=300, alias="CAPTURE_CHAIN_SECONDS")
    """Gap between recorded option chains. ~170 rows each."""
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    kill_switch_enabled: bool = Field(default=False, alias="KILL_SWITCH_ENABLED")

    bar_store_dir: str = Field(default="var/bars", alias="BAR_STORE_DIR")
    """Where observed bars are snapshotted so a restart does not lose them.

    They are expensive: NSE serves no history, so a week of 5-minute bars is
    a week of uptime. Empty disables persistence.
    """

    webhook_endpoints_file: str = Field(
        default="var/webhook-endpoints.json", alias="WEBHOOK_ENDPOINTS_FILE"
    )
    """The gateway's endpoint registry: slugs, their two credentials, and
    what each one does.

    A file rather than a database table, and edited on the machine rather
    than from the web page. Adding public ingress to a running system is an
    operator action — a page that mints new POST endpoints on demand is a
    page that can be used to mint them by whoever reaches it.
    """
    webhook_gateway_port: int = Field(default=8788, alias="WEBHOOK_GATEWAY_PORT")
    webhook_rate_limit: int = Field(default=120, alias="WEBHOOK_RATE_LIMIT")
    """Deliveries per minute per endpoint before the gateway answers 429.

    Retention already bounds the disk; this bounds the work. It is a
    setting rather than a constant because the right value depends on the
    sender — a 1-minute chart with six alert conditions is a different
    shape from a broker's order callbacks — and because a hardcoded limit
    cannot be exercised against a running process.
    """
    webhook_trust_forwarded_for: bool = Field(
        default=False, alias="WEBHOOK_TRUST_FORWARDED_FOR"
    )
    """Whether the gateway reads the source address from X-Forwarded-For.

    Required whenever a reverse proxy or tunnel is in front, which for
    TradingView it always is: TradingView calls port 80 or 443 only, so
    the gateway's own port is never the one being called. Without this,
    every request's peer address is the proxy and an endpoint's IP
    allowlist matches nothing — it rejects everything.

    Off by default because the header is client-supplied text: trusting it
    with nothing in front lets any sender name their own address.
    """
    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    """From @BotFather. Empty disables the chat bot entirely.

    A bearer credential: anyone holding it can act as the bot, so it goes
    in `.env` beside the broker tokens and never in the repository.
    """
    telegram_allowed_chat_ids: str = Field(
        default="", alias="TELEGRAM_ALLOWED_CHAT_IDS"
    )
    """Comma-separated numeric chat ids allowed to talk to the bot.

    Required, and the bot refuses to start without it. A bot with no
    allowlist answers anyone who finds it, and what it answers with is
    what your system is doing. Get your id by messaging the bot once and
    reading the log line it writes for the refusal.
    """
    console_base_url: str = Field(
        default="http://127.0.0.1:8000", alias="CONSOLE_BASE_URL"
    )
    """Where the read-only console API answers, for the chat bot to read.

    Loopback by default: the console is not on a public interface, and the
    bot runs on the same box. Reading it over HTTP rather than importing
    the engine is what keeps the bot unable to do anything but read.
    """

    signal_routes_file: str = Field(
        default="var/signal-routes.json", alias="SIGNAL_ROUTES_FILE"
    )
    """The relay's route table: where a strategy signal may go, and how far.

    Absent means no route exists and no signal can leave the machine, which
    is the correct default. Every route in it is additionally off until its
    own `enabled` is set — see `signals.relay`, which does not consult the
    brains, the Risk Engine or the Execution Gate.
    """
    """The gateway's own port, separate from the console's and the
    TradingView receiver's. All three have different exposure: the console
    is tailnet-only, the other two answer the public internet."""

    tradingview_webhook_secret: str = Field(default="", alias="TRADINGVIEW_WEBHOOK_SECRET")
    """The shared secret a TradingView alert must carry. Empty disables the
    receiver entirely.

    TradingView cannot sign a webhook, so this is the only credential the
    request carries and it travels in the body. It has no default on
    purpose: a receiver that starts with a well-known secret is an open
    endpoint on the public internet.
    """
    tradingview_webhook_port: int = Field(default=8787, alias="TRADINGVIEW_WEBHOOK_PORT")
    """The receiver's own port. Separate from the console's, because the
    console is served on the tailnet and this one has to be reachable from
    TradingView's servers."""
    tradingview_allowed_ips: str = Field(default="", alias="TRADINGVIEW_ALLOWED_IPS")
    """Comma-separated source addresses, or empty for TradingView's
    published egress set. The literal string `any` turns the address check
    off — needed behind a tunnel, where the peer address is the tunnel, and
    spelled out so it is a decision rather than an oversight."""
    tradingview_trust_forwarded_for: bool = Field(
        default=False, alias="TRADINGVIEW_TRUST_FORWARDED_FOR"
    )
    """Whether to read the source address from X-Forwarded-For. Only with a
    reverse proxy you control that overwrites the header — otherwise any
    sender can name their own address and the allowlist stops meaning
    anything."""


@lru_cache
def get_settings() -> Settings:
    return Settings()
