"""Spec §4. A trigger only means "something changed; analyze it" — it must
never directly create an order."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from index_option_brain.contracts.enums import TriggerType


class Event(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_id: str
    trigger_type: TriggerType
    timestamp: datetime
    payload: dict[str, Any] = Field(default_factory=dict)
    significance_score: float | None = None
