"""Spec §19-20. Feedback is recorded per-trade; a Lesson is a structured,
reviewable artifact — never an automatic mutation of live strategy parameters."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class TradeFeedback(BaseModel):
    model_config = ConfigDict(frozen=True)

    feedback_id: str
    thesis_id: str
    original_thesis: str
    scenario_id: str
    strategy: str
    strike_summary: str
    risk_summary: str
    expected_behavior: str
    actual_behavior: str
    exit_reason: str
    pnl: Decimal
    thesis_confirmed: bool
    failure_reason: str | None = None
    market_conditions: dict[str, str] = Field(default_factory=dict)
    recorded_at: datetime


class Lesson(BaseModel):
    model_config = ConfigDict(frozen=True)

    lesson_id: str
    derived_from_feedback_ids: list[str] = Field(default_factory=list)
    summary: str
    validated: bool = False
