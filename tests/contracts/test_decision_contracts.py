from decimal import Decimal

from index_option_brain.contracts.decision import TradeDecision
from index_option_brain.contracts.enums import Direction, StrategyType, TradeDecisionType
from index_option_brain.contracts.risk import RiskDecision


def test_no_trade_is_representable_without_a_risk_approval():
    """NO_TRADE / REJECT must always be constructible even when risk withheld
    approval — spec §12 ("NO TRADE must always be valid") and §14 ("If risk
    fails: REJECT")."""
    decision = TradeDecision(
        decision_id="d-1",
        state_id="s-1",
        thesis_id="t-1",
        decision=TradeDecisionType.REJECT,
        direction=Direction.NEUTRAL,
        strategy=StrategyType.NO_TRADE,
        confidence=0.0,
        max_loss=Decimal(0),
        risk_decision=RiskDecision(
            approved=False,
            reason_codes=["daily_loss_limit_reached"],
            max_loss=Decimal(0),
            quantity=0,
            exposure=Decimal(0),
            margin_required=Decimal(0),
        ),
    )
    assert decision.decision is TradeDecisionType.REJECT
    assert decision.risk_decision.approved is False


def test_strategy_type_has_no_trade_member():
    assert StrategyType.NO_TRADE in StrategyType
