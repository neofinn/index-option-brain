"""Spec §18, §30."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from index_option_brain.contracts.enums import TradeLifecycleState
from index_option_brain.contracts.instruments import OptionContractSpec


class PositionLeg(BaseModel):
    model_config = ConfigDict(frozen=True)

    contract: OptionContractSpec
    side: str  # "BUY" | "SELL"
    quantity: int
    average_price: Decimal


class Position(BaseModel):
    model_config = ConfigDict(frozen=True)

    position_id: str
    thesis_id: str
    state: TradeLifecycleState
    legs: list[PositionLeg] = Field(default_factory=list)
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    invalidation_conditions: list[str] = Field(default_factory=list)
    opened_at: datetime
    updated_at: datetime


class PositionEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    position_event_id: str
    position_id: str
    timestamp: datetime
    from_state: TradeLifecycleState | None
    to_state: TradeLifecycleState
    detail: dict[str, str] = Field(default_factory=dict)


class PositionState(BaseModel):
    """The slice of MarketState (spec §3) describing currently held / watched
    positions."""

    model_config = ConfigDict(frozen=True)

    positions: list[Position] = Field(default_factory=list)
