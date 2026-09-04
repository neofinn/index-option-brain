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


@lru_cache
def get_settings() -> Settings:
    return Settings()
