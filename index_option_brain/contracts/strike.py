"""Spec §13."""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from index_option_brain.contracts.instruments import OptionContractSpec


class StrikeCandidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    contract: OptionContractSpec
    score: float
    delta: Decimal
    liquidity_score: float
    bid_ask_spread: Decimal
    capital_required: Decimal
    max_loss: Decimal
    rationale: str
