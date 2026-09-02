"""Spec §14. The Risk Engine has absolute authority over trade authorization.

If risk fails, or cannot be evaluated, the answer is REJECT. No AI/agent may
override a RiskDecision (spec §23, §35) — and nothing in this module gives one
a way to: `AgentAssessment` cannot produce a `RiskDecision`, and the Execution
Gate reads `approved` rather than re-deriving it.

Reason codes are an enum rather than free text. They are read by the Execution
Gate, counted by the observability layer (spec §31 "risk failures"), and
asserted on in tests — none of which survives a hand-written string.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from index_option_brain.contracts.enums import Direction, StrategyType
from index_option_brain.contracts.instruments import AccountSnapshot
from index_option_brain.contracts.position import Position
from index_option_brain.contracts.strike import StrikeCandidate


class RiskReasonCode(StrEnum):
    """Why risk decided what it decided. Every rejection carries at least one."""

    APPROVED = "APPROVED"

    # Budget and sizing
    INSUFFICIENT_RISK_BUDGET = "INSUFFICIENT_RISK_BUDGET"
    INSUFFICIENT_MARGIN = "INSUFFICIENT_MARGIN"
    MAX_LOSS_ABOVE_CEILING = "MAX_LOSS_ABOVE_CEILING"
    BELOW_MINIMUM_SIZE = "BELOW_MINIMUM_SIZE"

    # Portfolio limits
    DAILY_LOSS_LIMIT_REACHED = "DAILY_LOSS_LIMIT_REACHED"
    MAX_POSITIONS_REACHED = "MAX_POSITIONS_REACHED"
    STRATEGY_LIMIT_REACHED = "STRATEGY_LIMIT_REACHED"
    INSTRUMENT_LIMIT_REACHED = "INSTRUMENT_LIMIT_REACHED"
    EXPOSURE_LIMIT_REACHED = "EXPOSURE_LIMIT_REACHED"
    CONCENTRATION_LIMIT_REACHED = "CONCENTRATION_LIMIT_REACHED"

    # Market quality
    LIQUIDITY_BELOW_FLOOR = "LIQUIDITY_BELOW_FLOOR"
    SLIPPAGE_ABOVE_CEILING = "SLIPPAGE_ABOVE_CEILING"

    # Structure
    UNDEFINED_RISK_STRUCTURE = "UNDEFINED_RISK_STRUCTURE"

    # Context
    EVENT_RISK_BLACKOUT = "EVENT_RISK_BLACKOUT"

    # Fail-closed
    EVALUATION_FAILED = "EVALUATION_FAILED"


class ScheduledEvent(BaseModel):
    """A known event whose timing alone changes the risk of holding a position
    — RBI policy, the Union Budget, an index rebalance (spec §4, §14)."""

    model_config = ConfigDict(frozen=True)

    name: str
    starts_at: datetime
    blocks_new_entries: bool = True


class PortfolioState(BaseModel):
    """Everything the Risk Engine must weigh beyond the candidate itself.

    Carried as one typed object rather than as loose arguments, so a new limit
    cannot be added without the state it needs being visible on the contract
    (spec §3: no uncontrolled variables between modules).
    """

    model_config = ConfigDict(frozen=True)

    account: AccountSnapshot
    open_positions: list[Position] = Field(default_factory=list)
    daily_realized_pnl: Decimal = Decimal(0)
    """Signed: negative is a loss taken today."""
    daily_unrealized_pnl: Decimal = Decimal(0)
    committed_margin: Decimal = Decimal(0)
    scheduled_events: list[ScheduledEvent] = Field(default_factory=list)

    @property
    def open_position_count(self) -> int:
        return len(self.open_positions)

    @property
    def day_pnl(self) -> Decimal:
        return self.daily_realized_pnl + self.daily_unrealized_pnl

    def count_for_strategy(self, strategy: StrategyType) -> int:
        return sum(1 for p in self.open_positions if p.strategy is strategy)

    def count_for_underlying(self, symbol: str) -> int:
        return sum(
            1
            for p in self.open_positions
            if any(leg.contract.underlying_symbol == symbol for leg in p.legs)
        )

    def exposure_for_underlying(self, symbol: str) -> Decimal:
        """Committed max loss against one underlying — the concentration
        measure that matters for defined-risk options."""
        return sum(
            (
                p.max_loss
                for p in self.open_positions
                if any(leg.contract.underlying_symbol == symbol for leg in p.legs)
            ),
            Decimal(0),
        )

    @property
    def total_exposure(self) -> Decimal:
        return sum((p.max_loss for p in self.open_positions), Decimal(0))


class TradeCandidate(BaseModel):
    """Spec §33's TRADE CANDIDATE: the ranked structure plus the thesis that
    produced it, presented to risk for authorization.

    This type is what breaks the circularity in the spec's own contracts — a
    `TradeDecision` requires a `RiskDecision`, and a `RiskDecision` requires
    something to authorize. That something is this, not a half-built
    TradeDecision with a placeholder approval in it.
    """

    model_config = ConfigDict(frozen=True)

    state_id: str
    thesis_id: str
    signal_id: str
    scenario_id: str
    direction: Direction
    strategy: StrategyType
    structure: StrikeCandidate
    """Priced for a single lot. The Risk Engine decides how many."""
    underlying_symbol: str
    confidence: float
    entry_conditions: list[str] = Field(default_factory=list)
    target_conditions: list[str] = Field(default_factory=list)
    invalidation_conditions: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class RiskDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    approved: bool
    reason_codes: list[RiskReasonCode] = Field(default_factory=list)
    max_loss: Decimal
    """Sized to the authorized quantity, not the per-lot figure."""
    quantity: int
    """Total contracts (lots x lot size), matching broker order semantics."""
    lots: int = 0
    exposure: Decimal = Decimal(0)
    margin_required: Decimal = Decimal(0)
    evidence: list[str] = Field(default_factory=list)

    @property
    def rejected(self) -> bool:
        return not self.approved

    @classmethod
    def reject(
        cls, *codes: RiskReasonCode, evidence: list[str] | None = None
    ) -> RiskDecision:
        """A rejection is always fully zeroed. There is no such thing as a
        partially-approved trade carrying a live size."""
        return cls(
            approved=False,
            reason_codes=list(codes),
            max_loss=Decimal(0),
            quantity=0,
            lots=0,
            exposure=Decimal(0),
            margin_required=Decimal(0),
            evidence=evidence or [],
        )
