"""Spec §18, §30.

`thesis_direction`, `max_loss`, and `target_profit` are stored on the
Position itself so the Position Brain can answer "is the reason for entering
the trade still valid?" mechanically, rather than by re-parsing prose. The
free-text `invalidation_conditions` remain for auditability and reviewer
context.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from index_option_brain.contracts.enums import (
    Direction,
    OrderSide,
    StrategyType,
    TradeLifecycleState,
)
from index_option_brain.contracts.instruments import Greeks, OptionContractSpec


class PositionLeg(BaseModel):
    model_config = ConfigDict(frozen=True)

    contract: OptionContractSpec
    side: OrderSide
    quantity: int
    """Contract **units**, not lots.

    Stated because `StrikeLeg.lots` — the contract this one is built from —
    counts lots, and reading one as the other is a `lot_size` error (65x on
    NIFTY) that produces a plausible number rather than an obvious one.
    `analytics.exposure.leg_exposure` refuses to guess between them.
    """
    average_price: Decimal
    current_price: Decimal | None = None
    greeks: Greeks | None = None
    """Greeks as of the last mark, or None when they could not be computed.

    None rather than zero: a delta of zero is a hedged leg, an absent delta
    is an unmeasured one, and portfolio exposure reports the difference
    rather than letting an invisible leg pass a limit.
    """

    @property
    def signed_quantity(self) -> int:
        return self.quantity if self.side is OrderSide.BUY else -self.quantity

    def unrealized_pnl(self) -> Decimal:
        if self.current_price is None:
            return Decimal(0)
        return (self.current_price - self.average_price) * self.signed_quantity


class Position(BaseModel):
    model_config = ConfigDict(frozen=True)

    position_id: str
    thesis_id: str
    state: TradeLifecycleState
    strategy: StrategyType
    thesis_direction: Direction
    legs: list[PositionLeg] = Field(default_factory=list)
    unrealized_pnl: Decimal = Decimal(0)
    realized_pnl: Decimal = Decimal(0)
    max_loss: Decimal = Decimal(0)
    target_profit: Decimal | None = None
    invalidation_conditions: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    opened_at: datetime
    updated_at: datetime

    @property
    def is_open(self) -> bool:
        return self.state not in (
            TradeLifecycleState.CLOSED,
            TradeLifecycleState.RECONCILED,
        )


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

    @property
    def open_positions(self) -> list[Position]:
        return [p for p in self.positions if p.is_open]

    @property
    def total_unrealized_pnl(self) -> Decimal:
        return sum((p.unrealized_pnl for p in self.positions), Decimal(0))
