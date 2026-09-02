"""Spec §12. NO_TRADE must always be a valid StrategyCandidate."""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from index_option_brain.contracts.enums import StrategyType


class StrategyCandidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    strategy: StrategyType
    score: float
    max_loss: Decimal
    max_profit: Decimal | None  # None permitted only for undefined-risk theoretical upside
    breakeven: list[Decimal]
    rationale: str
