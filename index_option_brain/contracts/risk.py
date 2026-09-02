"""Spec §14. The Risk Engine has absolute authority over trade authorization —
no AI/agent may override a RiskDecision (spec §23, §35)."""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class RiskDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    approved: bool
    reason_codes: list[str] = Field(default_factory=list)
    max_loss: Decimal
    quantity: int
    exposure: Decimal
    margin_required: Decimal
