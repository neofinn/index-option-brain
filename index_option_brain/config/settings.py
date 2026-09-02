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

    database_url: str = Field(
        default="postgresql+asyncpg://localhost:5432/index_option_brain",
        alias="DATABASE_URL",
    )
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
