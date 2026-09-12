"""Spec §15. The thesis_id persists through the entire trade lifecycle,
linking a TradeDecision to its later position, feedback, and lessons.

A TradeDecision is a *decision*, not an order. EXECUTE means the analysis
concluded and risk authorized; it does not mean anything was sent. Only the
Execution Gate (§16) and Order Manager (§17) turn this into an order, and the
gate re-validates independently rather than trusting this object.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from index_option_brain.contracts.enums import Direction, StrategyType, TradeDecisionType
from index_option_brain.contracts.risk import RiskDecision, TradeCandidate
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

    signal_id: str | None = None
    scenario_id: str | None = None
    underlying_symbol: str | None = None
    created_at: datetime | None = None

    @property
    def is_executable(self) -> bool:
        """EXECUTE *and* risk-approved. Both, because a decision is only ever
        as good as the authorization attached to it."""
        return (
            self.decision is TradeDecisionType.EXECUTE
            and self.risk_decision.approved
            and self.risk_decision.quantity > 0
        )

    @classmethod
    def from_candidate(
        cls,
        candidate: TradeCandidate,
        risk_decision: RiskDecision,
        *,
        timestamp: datetime,
        extra_evidence: list[str] | None = None,
    ) -> TradeDecision:
        """Assemble the decision from an authorized (or rejected) candidate.

        The decision type follows risk, never the other way round: risk
        approving yields EXECUTE, risk declining yields REJECT. There is no
        path here that produces EXECUTE without an approval.
        """
        decision = (
            TradeDecisionType.EXECUTE if risk_decision.approved else TradeDecisionType.REJECT
        )
        return cls(
            decision_id=uuid.uuid4().hex[:12],
            state_id=candidate.state_id,
            thesis_id=candidate.thesis_id,
            decision=decision,
            direction=candidate.direction,
            strategy=candidate.strategy,
            contracts=[candidate.structure] if risk_decision.approved else [],
            entry_conditions=list(candidate.entry_conditions),
            target_conditions=list(candidate.target_conditions),
            invalidation_conditions=list(candidate.invalidation_conditions),
            confidence=candidate.confidence,
            max_loss=risk_decision.max_loss,
            evidence=[*candidate.evidence, *risk_decision.evidence, *(extra_evidence or [])],
            risk_decision=risk_decision,
            signal_id=candidate.signal_id,
            scenario_id=candidate.scenario_id,
            underlying_symbol=candidate.underlying_symbol,
            created_at=timestamp,
        )

    @classmethod
    def wait(
        cls,
        *,
        state_id: str,
        thesis_id: str | None = None,
        reasons: list[str],
        timestamp: datetime,
        direction: Direction = Direction.NEUTRAL,
    ) -> TradeDecision:
        """No candidate reached risk. WAIT is a real decision with reasons —
        the absence of a trade is recorded, not merely unlogged."""
        return cls(
            decision_id=uuid.uuid4().hex[:12],
            state_id=state_id,
            thesis_id=thesis_id or uuid.uuid4().hex[:12],
            decision=TradeDecisionType.WAIT,
            direction=direction,
            strategy=StrategyType.NO_TRADE,
            confidence=0.0,
            max_loss=Decimal(0),
            evidence=reasons,
            risk_decision=RiskDecision.reject(),
            created_at=timestamp,
        )
