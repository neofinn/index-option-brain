"""Spec §15. The thesis_id persists through the entire trade lifecycle,
linking a TradeDecision to its later position, feedback, and lessons."""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from index_option_brain.contracts.enums import Direction, StrategyType, TradeDecisionType
from index_option_brain.contracts.risk import RiskDecision
from index_option_brain.contracts.strike import StrikeCandidate


class TradeDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    decision_id: str
    state_id: str
    thesis_id: str
    decision: TradeDecisionType
    direction: Direction
    strategy: StrategyType
    contracts: list[StrikeCandidate] = Field(default_factory=list)
    entry_conditions: list[str] = Field(default_factory=list)
    target_conditions: list[str] = Field(default_factory=list)
    invalidation_conditions: list[str] = Field(default_factory=list)
    confidence: float
    max_loss: Decimal
    evidence: list[str] = Field(default_factory=list)
    risk_decision: RiskDecision
