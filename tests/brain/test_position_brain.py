"""Position Brain behaviour (spec §18).

The question under test is "is the reason for entering still valid?", which
is deliberately not "am I in profit". A position can be green with a dead
thesis and red with an intact one, so these tests separate the two.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from index_option_brain.brain.pipeline import QuantitativeBrain
from index_option_brain.brain.position_brain import DeterministicPositionBrain
from index_option_brain.contracts.enums import (
    Direction,
    OrderSide,
    StrategyType,
    TradeLifecycleState,
)
from index_option_brain.contracts.market_state import MarketState
from index_option_brain.contracts.position import Position, PositionLeg, PositionState

brain = DeterministicPositionBrain()


def analysed(state: MarketState) -> MarketState:
    return QuantitativeBrain().run(state).state


def position_in(
    state: MarketState,
    *,
    thesis_direction: Direction = Direction.BULLISH,
    entry_multiplier: Decimal = Decimal("1.0"),
    max_loss: Decimal = Decimal(50000),
    target_profit: Decimal | None = None,
    lifecycle: TradeLifecycleState = TradeLifecycleState.ACTIVE,
) -> Position:
    """A long call on the ATM strike, entered at `entry_multiplier` x its
    current price — so the caller controls whether it is winning or losing."""
    spot = state.spot
    atm = min(
        (q for q in state.options_state.chain if q.contract.option_type.value == "CE"),
        key=lambda q: abs(q.contract.strike - spot),
    )
    return Position(
        position_id="p1",
        thesis_id="t1",
        state=lifecycle,
        strategy=StrategyType.LONG_CALL,
        thesis_direction=thesis_direction,
        legs=[
            PositionLeg(
                contract=atm.contract,
                side=OrderSide.BUY,
                quantity=atm.contract.lot_size,
                average_price=atm.ltp * entry_multiplier,
            )
        ],
        max_loss=max_loss,
        target_profit=target_profit,
        opened_at=datetime(2026, 9, 4, 4, 0, tzinfo=UTC),
        updated_at=datetime(2026, 9, 4, 4, 0, tzinfo=UTC),
    )


class TestRepricing:
    def test_legs_are_marked_at_exit_prices(self, uptrend_state: MarketState):
        """A long is closed into the bid, not at the last traded price."""
        state = analysed(uptrend_state)
        updated = brain.evaluate(position_in(state), state)
        leg = updated.legs[0]
        quote = next(
            q
            for q in state.options_state.chain
            if q.contract.strike == leg.contract.strike
            and q.contract.option_type is leg.contract.option_type
        )
        assert leg.current_price == quote.bid

    def test_a_winning_position_reports_positive_pnl(self, uptrend_state: MarketState):
        state = analysed(uptrend_state)
        updated = brain.evaluate(position_in(state, entry_multiplier=Decimal("0.5")), state)
        assert updated.unrealized_pnl > 0

    def test_a_losing_position_reports_negative_pnl(self, uptrend_state: MarketState):
        state = analysed(uptrend_state)
        updated = brain.evaluate(position_in(state, entry_multiplier=Decimal("2.0")), state)
        assert updated.unrealized_pnl < 0

    def test_a_leg_absent_from_the_chain_is_left_unmarked(self, uptrend_state: MarketState):
        state = analysed(uptrend_state)
        empty_chain = state.options_state.model_copy(update={"chain": []})
        updated = brain.evaluate(
            position_in(state), state.model_copy(update={"options_state": empty_chain})
        )
        assert updated.legs[0].current_price is None


class TestThesisValidity:
    def test_an_intact_thesis_keeps_the_position_active(self, uptrend_state: MarketState):
        state = analysed(uptrend_state)
        updated = brain.evaluate(
            position_in(state, thesis_direction=Direction.BULLISH), state
        )
        assert updated.state is TradeLifecycleState.ACTIVE
        assert any("still supports the thesis" in item for item in updated.evidence)

    def test_a_broken_thesis_requests_an_exit(self, downtrend_state: MarketState):
        """The market now opposes the direction the trade was entered on."""
        state = analysed(downtrend_state)
        updated = brain.evaluate(
            position_in(state, thesis_direction=Direction.BULLISH), state
        )
        assert updated.state is TradeLifecycleState.EXIT_PENDING
        assert any("thesis invalidated" in item for item in updated.evidence)

    def test_a_profitable_position_with_a_dead_thesis_still_exits(
        self, downtrend_state: MarketState
    ):
        """Profit does not validate a thesis. This is the case a P&L-driven
        exit rule gets wrong."""
        state = analysed(downtrend_state)
        updated = brain.evaluate(
            position_in(
                state, thesis_direction=Direction.BULLISH, entry_multiplier=Decimal("0.2")
            ),
            state,
        )
        assert updated.unrealized_pnl > 0
        assert updated.state is TradeLifecycleState.EXIT_PENDING

    def test_without_analysis_the_thesis_cannot_be_reassessed(
        self, uptrend_state: MarketState
    ):
        state = uptrend_state.model_copy(update={"analysis": None})
        updated = brain.evaluate(position_in(state), state)
        assert any("cannot be reassessed" in item for item in updated.evidence)
        assert updated.state is not TradeLifecycleState.EXIT_PENDING


class TestRiskAndTargetExits:
    def test_approaching_the_authorized_max_loss_requests_an_exit(
        self, uptrend_state: MarketState
    ):
        state = analysed(uptrend_state)
        position = position_in(
            state, entry_multiplier=Decimal("3.0"), max_loss=Decimal(20000)
        )
        updated = brain.evaluate(position, state)
        assert updated.unrealized_pnl < 0
        assert updated.state is TradeLifecycleState.EXIT_PENDING
        assert any("max loss" in item for item in updated.evidence)

    def test_reaching_the_target_requests_an_exit(self, uptrend_state: MarketState):
        state = analysed(uptrend_state)
        position = position_in(
            state, entry_multiplier=Decimal("0.5"), target_profit=Decimal(1)
        )
        updated = brain.evaluate(position, state)
        assert updated.state is TradeLifecycleState.EXIT_PENDING
        assert any("target" in item for item in updated.evidence)

    def test_a_modest_drawdown_is_reported_without_exiting(
        self, uptrend_state: MarketState
    ):
        state = analysed(uptrend_state)
        position = position_in(
            state, entry_multiplier=Decimal("1.05"), max_loss=Decimal(500000)
        )
        updated = brain.evaluate(position, state)
        assert updated.state is not TradeLifecycleState.EXIT_PENDING
        assert any("Drawdown" in item for item in updated.evidence)


class TestExitability:
    def test_collapsed_liquidity_requests_an_exit(self, uptrend_state: MarketState):
        """A position that cannot be exited is a different risk from one that
        is merely losing."""
        state = analysed(uptrend_state)
        widened = [
            q.model_copy(update={"bid": q.ltp * Decimal("0.5"), "ask": q.ltp * Decimal("1.5")})
            for q in state.options_state.chain
        ]
        wide_state = state.model_copy(
            update={"options_state": state.options_state.model_copy(update={"chain": widened})}
        )
        updated = brain.evaluate(position_in(state), wide_state)
        assert updated.state is TradeLifecycleState.EXIT_PENDING
        assert any("liquidity deteriorated" in item for item in updated.evidence)

    def test_imminent_expiry_requests_an_exit(self, expiry_day_state: MarketState):
        state = analysed(expiry_day_state)
        near_expiry = state.volatility_state.model_copy(update={"days_to_expiry": 0.1})
        updated = brain.evaluate(
            position_in(state), state.model_copy(update={"volatility_state": near_expiry})
        )
        assert updated.state is TradeLifecycleState.EXIT_PENDING
        assert any("expiry" in item for item in updated.evidence)


class TestLifecycle:
    def test_closed_positions_are_left_untouched(self, uptrend_state: MarketState):
        state = analysed(uptrend_state)
        closed = position_in(state, lifecycle=TradeLifecycleState.CLOSED)
        assert brain.evaluate(closed, state) is closed

    def test_the_brain_never_returns_an_order(self, uptrend_state: MarketState):
        """Spec §18: it may request a state change, never place an order."""
        state = analysed(uptrend_state)
        updated = brain.evaluate(position_in(state), state)
        assert isinstance(updated, Position)
        assert not hasattr(updated, "order_id")

    def test_updated_at_comes_from_the_market_state_not_the_wall_clock(
        self, uptrend_state: MarketState
    ):
        """Spec §22 runs the same brain in BACKTEST and REPLAY, where the wall
        clock would date historical positions to today and make the audit
        trail non-reproducible."""
        state = analysed(uptrend_state)
        position = position_in(state)
        updated = brain.evaluate(position, state)
        assert updated.updated_at == state.timestamp
        assert updated.updated_at > position.updated_at

    @pytest.mark.parametrize(
        "direction", [Direction.BULLISH, Direction.BEARISH, Direction.NEUTRAL]
    )
    def test_every_thesis_direction_is_evaluable(
        self, uptrend_state: MarketState, direction: Direction
    ):
        state = analysed(uptrend_state)
        updated = brain.evaluate(position_in(state, thesis_direction=direction), state)
        assert updated.evidence


class TestPipelineIntegration:
    def test_the_pipeline_evaluates_open_positions(self, uptrend_state: MarketState):
        state = analysed(uptrend_state)
        with_position = state.with_position_state(
            PositionState(positions=[position_in(state)])
        )
        result = QuantitativeBrain().run(with_position)
        assert len(result.positions) == 1
        assert result.positions[0].evidence
        assert result.state.position_state.positions[0].legs[0].current_price is not None
