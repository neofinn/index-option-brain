"""Spec §13.

A StrikeCandidate is a fully-specified, executable structure — not a single
option. A spread is two legs, and its economics (max loss, breakeven,
capital) are properties of the combination, so anything that priced legs
independently would misreport risk. Single-leg structures are simply
one-leg candidates.
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from index_option_brain.contracts.enums import OrderSide, StrategyType
from index_option_brain.contracts.instruments import OptionContractSpec


class StrikeLeg(BaseModel):
    model_config = ConfigDict(frozen=True)

    contract: OptionContractSpec
    side: OrderSide
    lots: int
    reference_price: Decimal
    delta: Decimal | None = None
    liquidity_score: float = 0.0


class StrikeCandidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    strategy: StrategyType
    legs: list[StrikeLeg]
    score: float
    net_premium: Decimal
    """Positive for a net debit paid, negative for a net credit received."""
    net_delta: Decimal
    liquidity_score: float
    worst_relative_spread: float
    capital_required: Decimal
    max_loss: Decimal
    max_profit: Decimal | None
    breakeven: list[Decimal] = Field(default_factory=list)
    rationale: str = ""

    @property
    def is_credit(self) -> bool:
        return self.net_premium < 0

    @property
    def reward_to_risk(self) -> float | None:
        if self.max_profit is None or self.max_loss <= 0:
            return None
        return float(self.max_profit / self.max_loss)
