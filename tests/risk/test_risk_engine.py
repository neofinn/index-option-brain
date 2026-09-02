"""Risk Engine behaviour (spec §14).

Spec §32 singles out risk and execution for especially strong coverage, so
this suite aims to make every reason code reachable, every limit bind in
isolation, and the fail-closed path provable rather than assumed.

The engine's job is not only to veto. It sizes — so most of what follows is
arithmetic on lot counts, checked against figures chosen to be verifiable by
eye (a per-lot max loss of ₹10,000 against equity of ₹20,00,000).
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest

from index_option_brain.contracts.enums import (
    Direction,
    OrderSide,
    StrategyType,
    TradeLifecycleState,
)
from index_option_brain.contracts.position import Position, PositionLeg
from index_option_brain.contracts.risk import (
    RiskDecision,
    RiskReasonCode,
    ScheduledEvent,
)
from index_option_brain.contracts.strike import StrikeCandidate
from index_option_brain.risk.limits import RiskLimits
from index_option_brain.risk.margin import DefinedRiskMarginModel, MarginModel
from index_option_brain.risk.risk_engine import DeterministicRiskEngine
from tests.risk.conftest import (
    LOT_SIZE,
    NOW,
    PER_LOT_MAX_LOSS,
    account,
    candidate,
    contract,
    portfolio,
)


class FlatMargin(MarginModel):
    """A predictable margin model, so margin never confounds a test that is
    about some other limit."""

    def __init__(self, per_lot: Decimal = Decimal(10000)) -> None:
        self.per_lot = per_lot

    def estimate(self, structure: StrikeCandidate, lots: int) -> Decimal:
        return self.per_lot * lots


def engine(**limit_overrides) -> DeterministicRiskEngine:
    return DeterministicRiskEngine(
        limits=RiskLimits(**limit_overrides), margin_model=FlatMargin()
    )


def open_position(
    *,
    strategy: StrategyType = StrategyType.PUT_CREDIT_SPREAD,
    max_loss: Decimal = Decimal(10000),
    underlying: str = "NIFTY",
    state: TradeLifecycleState = TradeLifecycleState.ACTIVE,
) -> Position:
    spec = contract(24500).model_copy(update={"underlying_symbol": underlying})
    return Position(
        position_id="p",
        thesis_id="t",
        state=state,
        strategy=strategy,
        thesis_direction=Direction.BULLISH,
        legs=[
            PositionLeg(
                contract=spec, side=OrderSide.SELL, quantity=LOT_SIZE,
                average_price=Decimal(90),
            )
        ],
        max_loss=max_loss,
        opened_at=NOW,
        updated_at=NOW,
    )


class TestApprovalAndSizing:
    def test_a_funded_account_is_approved_and_sized(self, trade, default_account, empty_portfolio):
        decision = engine().authorize(trade, default_account, empty_portfolio)
        assert decision.approved
        assert decision.reason_codes == [RiskReasonCode.APPROVED]
        # 1% of 20,00,000 is 20,000; at 10,000 per lot that is two lots.
        assert decision.lots == 2
        assert decision.quantity == 2 * LOT_SIZE
        assert decision.max_loss == PER_LOT_MAX_LOSS * 2
        assert decision.exposure == decision.max_loss
        assert decision.margin_required == Decimal(20000)

    def test_sizing_scales_with_the_risk_fraction(self, trade, default_account, empty_portfolio):
        for fraction, expected_lots in [("0.005", 1), ("0.01", 2), ("0.025", 5)]:
            decision = engine(max_risk_per_trade=Decimal(fraction)).authorize(
                trade, default_account, empty_portfolio
            )
            assert decision.lots == expected_lots, fraction

    def test_sizing_is_whole_lots_and_rounds_down(self, trade, empty_portfolio):
        # 1% of 25,00,000 is 25,000 — two and a half lots, so two.
        acct = account(equity="2500000")
        decision = engine().authorize(trade, acct, portfolio(acct=acct))
        assert decision.lots == 2

    def test_the_lot_cap_is_respected(self, trade, empty_portfolio):
        acct = account(equity="100000000", available_margin="100000000")
        decision = engine(max_lots=3).authorize(trade, acct, portfolio(acct=acct))
        assert decision.lots == 3
        assert "maximum lot cap" in " ".join(decision.evidence)

    def test_the_binding_constraint_is_named(self, trade, default_account, empty_portfolio):
        decision = engine().authorize(trade, default_account, empty_portfolio)
        assert any("Size limited by" in item for item in decision.evidence)

    def test_approval_reports_max_loss_as_a_share_of_equity(
        self, trade, default_account, empty_portfolio
    ):
        decision = engine().authorize(trade, default_account, empty_portfolio)
        assert any("% " in item or "%)" in item for item in decision.evidence)


class TestRejectionShape:
    def test_a_rejection_is_fully_zeroed(self, trade, empty_portfolio):
        """There is no such thing as a partially-approved trade carrying a
        live size."""
        acct = account(equity="100000")
        decision = engine().authorize(trade, acct, portfolio(acct=acct))
        assert decision.rejected
        assert decision.lots == 0
        assert decision.quantity == 0
        assert decision.max_loss == 0
        assert decision.exposure == 0
        assert decision.margin_required == 0

    def test_every_rejection_carries_at_least_one_reason(self, trade, empty_portfolio):
        acct = account(equity="100000")
        decision = engine().authorize(trade, acct, portfolio(acct=acct))
        assert decision.reason_codes
        assert RiskReasonCode.APPROVED not in decision.reason_codes

    def test_rejections_collect_every_applicable_reason(self, trade):
        """When a trade is refused, the useful question is what all was
        wrong — not which check happened to run first."""
        acct = account(equity="500000")
        state = portfolio(
            acct=acct,
            open_positions=[open_position(), open_position()],
            daily_realized_pnl=Decimal(-20000),
        )
        decision = engine().authorize(
            candidate(liquidity=0.1, spread=0.5), acct, state
        )
        assert {
            RiskReasonCode.DAILY_LOSS_LIMIT_REACHED,
            RiskReasonCode.LIQUIDITY_BELOW_FLOOR,
            RiskReasonCode.SLIPPAGE_ABOVE_CEILING,
            RiskReasonCode.STRATEGY_LIMIT_REACHED,
        } <= set(decision.reason_codes)


class TestPerTradeLimits:
    def test_a_per_lot_loss_above_the_absolute_ceiling_is_refused(
        self, default_account, empty_portfolio
    ):
        decision = engine(max_loss_ceiling=Decimal(5000)).authorize(
            candidate(), default_account, empty_portfolio
        )
        assert RiskReasonCode.MAX_LOSS_ABOVE_CEILING in decision.reason_codes

    def test_an_unsizeable_structure_is_refused(self, default_account, empty_portfolio):
        decision = engine().authorize(
            candidate(max_loss=Decimal(0)), default_account, empty_portfolio
        )
        assert RiskReasonCode.UNDEFINED_RISK_STRUCTURE in decision.reason_codes

    def test_long_premium_is_sized_on_its_bounded_loss(
        self, default_account, empty_portfolio
    ):
        """Unbounded profit is not unbounded risk — a long option has a max
        loss of the premium paid, so it sizes normally."""
        decision = engine().authorize(
            candidate(
                strategy=StrategyType.LONG_CALL,
                max_profit=None,
                net_premium=Decimal(10000),
            ),
            default_account,
            empty_portfolio,
        )
        assert decision.approved
        assert decision.lots == 2
        assert any("bounded side" in item for item in decision.evidence)

    def test_a_size_below_a_deliberate_minimum_is_distinguished(
        self, trade, default_account, empty_portfolio
    ):
        """Nothing fitting and what-fits-being-too-small are different
        failures."""
        decision = engine(min_lots=5).authorize(trade, default_account, empty_portfolio)
        assert decision.rejected
        assert decision.reason_codes[0] is RiskReasonCode.BELOW_MINIMUM_SIZE
        assert any("below the 5-lot minimum" in item for item in decision.evidence)


class TestMarketQualityLimits:
    def test_illiquid_structures_are_refused(self, default_account, empty_portfolio):
        decision = engine().authorize(
            candidate(liquidity=0.2), default_account, empty_portfolio
        )
        assert RiskReasonCode.LIQUIDITY_BELOW_FLOOR in decision.reason_codes

    def test_wide_spreads_are_refused_as_slippage(self, default_account, empty_portfolio):
        """The spread is the slippage control: it is paid entering and again
        exiting."""
        decision = engine().authorize(
            candidate(spread=0.12), default_account, empty_portfolio
        )
        assert RiskReasonCode.SLIPPAGE_ABOVE_CEILING in decision.reason_codes

    def test_a_structure_at_the_liquidity_floor_is_allowed(
        self, default_account, empty_portfolio
    ):
        decision = engine(min_liquidity_score=0.4).authorize(
            candidate(liquidity=0.4), default_account, empty_portfolio
        )
        assert decision.approved


class TestDailyLoss:
    def test_the_daily_loss_limit_blocks_new_entries(self, trade, default_account):
        # 3% of 20,00,000 is 60,000.
        state = portfolio(acct=default_account, daily_realized_pnl=Decimal(-60000))
        decision = engine().authorize(trade, default_account, state)
        assert RiskReasonCode.DAILY_LOSS_LIMIT_REACHED in decision.reason_codes

    def test_unrealized_losses_count_toward_the_daily_limit(self, trade, default_account):
        state = portfolio(
            acct=default_account,
            daily_realized_pnl=Decimal(-30000),
            daily_unrealized_pnl=Decimal(-30000),
        )
        decision = engine().authorize(trade, default_account, state)
        assert RiskReasonCode.DAILY_LOSS_LIMIT_REACHED in decision.reason_codes

    def test_a_partial_daily_loss_shrinks_the_next_position(self, trade, default_account):
        """One more trade must not be able to breach the daily limit, so the
        remaining allowance caps the size."""
        state = portfolio(acct=default_account, daily_realized_pnl=Decimal(-50000))
        decision = engine().authorize(trade, default_account, state)
        # 60,000 allowance less 50,000 lost leaves 10,000 — one lot.
        assert decision.approved
        assert decision.lots == 1

    def test_a_profitable_day_does_not_inflate_the_per_trade_budget(
        self, trade, default_account
    ):
        state = portfolio(acct=default_account, daily_realized_pnl=Decimal(500000))
        decision = engine().authorize(trade, default_account, state)
        assert decision.lots == 2, "the per-trade risk fraction still binds"


class TestPortfolioLimits:
    def test_the_open_position_count_is_capped(self, trade, default_account):
        state = portfolio(
            acct=default_account,
            open_positions=[
                open_position(strategy=StrategyType.LONG_CALL),
                open_position(strategy=StrategyType.LONG_PUT),
            ],
        )
        decision = engine(max_open_positions=2).authorize(trade, default_account, state)
        assert RiskReasonCode.MAX_POSITIONS_REACHED in decision.reason_codes

    def test_positions_per_strategy_are_capped(self, trade, default_account):
        state = portfolio(
            acct=default_account,
            open_positions=[open_position(), open_position()],
        )
        decision = engine(max_positions_per_strategy=2).authorize(
            trade, default_account, state
        )
        assert RiskReasonCode.STRATEGY_LIMIT_REACHED in decision.reason_codes

    def test_positions_per_underlying_are_capped(self, trade, default_account):
        state = portfolio(
            acct=default_account,
            open_positions=[
                open_position(strategy=StrategyType.LONG_CALL),
                open_position(strategy=StrategyType.LONG_PUT),
            ],
        )
        decision = engine(
            max_positions_per_underlying=2, max_open_positions=9
        ).authorize(trade, default_account, state)
        assert RiskReasonCode.INSTRUMENT_LIMIT_REACHED in decision.reason_codes

    def test_a_position_on_another_underlying_does_not_count(
        self, trade, default_account
    ):
        state = portfolio(
            acct=default_account,
            open_positions=[
                open_position(underlying="BANKNIFTY", strategy=StrategyType.LONG_CALL)
            ],
        )
        decision = engine(max_positions_per_underlying=1).authorize(
            trade, default_account, state
        )
        assert decision.approved

    def test_portfolio_exposure_caps_the_size(self, trade, default_account):
        """6% of 20,00,000 is 1,20,000 of committed max loss; 1,10,000 is
        already committed, leaving room for one lot."""
        state = portfolio(
            acct=default_account,
            open_positions=[open_position(max_loss=Decimal(110000))],
        )
        decision = engine(max_concentration_per_underlying=Decimal(1)).authorize(
            trade, default_account, state
        )
        assert decision.approved
        assert decision.lots == 1

    def test_exhausted_portfolio_exposure_refuses_the_trade(self, trade, default_account):
        state = portfolio(
            acct=default_account,
            open_positions=[open_position(max_loss=Decimal(120000))],
        )
        decision = engine(max_concentration_per_underlying=Decimal(1)).authorize(
            trade, default_account, state
        )
        assert RiskReasonCode.EXPOSURE_LIMIT_REACHED in decision.reason_codes

    def test_concentration_is_measured_per_underlying(self, trade, default_account):
        """4% of 20,00,000 is 80,000 against NIFTY specifically."""
        state = portfolio(
            acct=default_account,
            open_positions=[open_position(max_loss=Decimal(80000))],
        )
        decision = engine().authorize(trade, default_account, state)
        assert RiskReasonCode.CONCENTRATION_LIMIT_REACHED in decision.reason_codes

    def test_concentration_ignores_other_underlyings(self, trade, default_account):
        state = portfolio(
            acct=default_account,
            open_positions=[
                open_position(
                    underlying="BANKNIFTY",
                    max_loss=Decimal(80000),
                    strategy=StrategyType.LONG_CALL,
                )
            ],
        )
        decision = engine().authorize(trade, default_account, state)
        assert RiskReasonCode.CONCENTRATION_LIMIT_REACHED not in decision.reason_codes


class TestMargin:
    def test_insufficient_margin_refuses_the_trade(self, trade, empty_portfolio):
        acct = account(equity="2000000", available_margin="5000")
        decision = engine().authorize(trade, acct, portfolio(acct=acct))
        assert RiskReasonCode.INSUFFICIENT_MARGIN in decision.reason_codes

    def test_margin_utilization_caps_the_size(self, trade):
        """Only 60% of available margin may be committed: 60% of 25,000 is
        15,000, and one lot needs 10,000."""
        acct = account(equity="2000000", available_margin="25000")
        decision = engine().authorize(trade, acct, portfolio(acct=acct))
        assert decision.approved
        assert decision.lots == 1
        assert RiskReasonCode.APPROVED in decision.reason_codes

    def test_the_reported_margin_matches_the_authorized_size(
        self, trade, default_account, empty_portfolio
    ):
        decision = engine().authorize(trade, default_account, empty_portfolio)
        assert decision.margin_required == Decimal(10000) * decision.lots

    def test_the_default_margin_model_charges_the_debit_on_long_premium(self):
        model = DefinedRiskMarginModel()
        from tests.risk.conftest import structure

        long_call = structure(net_premium=Decimal(8000), max_loss=Decimal(8000))
        assert model.estimate(long_call, 2) == Decimal(16000)

    def test_the_default_margin_model_buffers_a_credit_spread(self):
        """The real figure moves with volatility and time, so the estimate
        errs high — an under-estimate produces a broker rejection at the
        worst possible moment."""
        model = DefinedRiskMarginModel()
        from tests.risk.conftest import structure

        spread = structure(net_premium=Decimal(-2500), max_loss=Decimal(10000))
        assert model.estimate(spread, 1) > Decimal(10000)

    def test_zero_lots_needs_no_margin(self):
        from tests.risk.conftest import structure

        assert DefinedRiskMarginModel().estimate(structure(), 0) == Decimal(0)


class TestEventRisk:
    def test_an_event_inside_the_blackout_blocks_entry(self, trade, default_account):
        state = portfolio(
            acct=default_account,
            scheduled_events=[
                ScheduledEvent(name="RBI policy", starts_at=NOW + timedelta(hours=6))
            ],
        )
        decision = engine().authorize(trade, default_account, state)
        assert RiskReasonCode.EVENT_RISK_BLACKOUT in decision.reason_codes
        assert any("RBI policy" in item for item in decision.evidence)

    def test_an_event_beyond_the_blackout_does_not_block(self, trade, default_account):
        state = portfolio(
            acct=default_account,
            scheduled_events=[
                ScheduledEvent(name="Union Budget", starts_at=NOW + timedelta(days=9))
            ],
        )
        assert engine().authorize(trade, default_account, state).approved

    def test_a_non_blocking_event_is_ignored(self, trade, default_account):
        state = portfolio(
            acct=default_account,
            scheduled_events=[
                ScheduledEvent(
                    name="Index rebalance",
                    starts_at=NOW + timedelta(hours=2),
                    blocks_new_entries=False,
                )
            ],
        )
        assert engine().authorize(trade, default_account, state).approved

    def test_a_past_event_is_ignored(self, trade, default_account):
        state = portfolio(
            acct=default_account,
            scheduled_events=[
                ScheduledEvent(name="Yesterday's policy", starts_at=NOW - timedelta(days=1))
            ],
        )
        assert engine().authorize(trade, default_account, state).approved

    def test_blackout_is_measured_from_the_account_snapshot_not_today(self, trade):
        """A replayed decision must black out against the events of *that*
        day, which is why the reference time comes from the snapshot."""
        historical = account()
        state = portfolio(
            acct=historical,
            scheduled_events=[
                ScheduledEvent(name="RBI policy", starts_at=NOW + timedelta(hours=3))
            ],
        )
        decision = engine().authorize(trade, historical, state)
        assert RiskReasonCode.EVENT_RISK_BLACKOUT in decision.reason_codes


class TestFailClosed:
    def test_an_exception_during_evaluation_rejects(self, trade, default_account, empty_portfolio):
        """Spec §29: if the risk engine fails, the answer is NO_TRADE. A bug
        in a limit check must never read as an approval."""

        class ExplodingMargin(MarginModel):
            def estimate(self, structure: StrikeCandidate, lots: int) -> Decimal:
                raise RuntimeError("margin service unreachable")

        risky = DeterministicRiskEngine(margin_model=ExplodingMargin())
        decision = risky.authorize(trade, default_account, empty_portfolio)

        assert decision.rejected
        assert decision.reason_codes == [RiskReasonCode.EVALUATION_FAILED]
        assert decision.quantity == 0
        assert any("RuntimeError" in item for item in decision.evidence)

    def test_zero_equity_is_refused(self, trade, empty_portfolio):
        acct = account(equity="0")
        decision = engine().authorize(trade, acct, portfolio(acct=acct))
        assert decision.rejected
        assert RiskReasonCode.INSUFFICIENT_RISK_BUDGET in decision.reason_codes

    def test_negative_equity_is_refused(self, trade):
        acct = account(equity="-50000")
        decision = engine().authorize(trade, acct, portfolio(acct=acct))
        assert decision.rejected


class TestNoOverride:
    def test_authorize_takes_no_override_parameter(self):
        """Spec §23/§35: no AI/agent may override risk. The absence of a
        force/override argument is the structural guarantee."""
        import inspect

        params = set(inspect.signature(DeterministicRiskEngine.authorize).parameters)
        assert params == {"self", "trade", "account", "portfolio"}

    def test_the_agent_layer_cannot_reach_the_risk_engine(self):
        """An import-level check, because a code path that does not exist
        cannot be exercised by a test that calls it."""
        import pathlib

        agent_dir = pathlib.Path(__file__).resolve().parents[2] / "index_option_brain" / "agent"
        sources = list(agent_dir.glob("*.py"))
        assert sources, "expected the agent package to exist"
        for path in sources:
            text = path.read_text()
            assert "index_option_brain.risk" not in text, path.name
            assert "index_option_brain.execution" not in text, path.name

    def test_confidence_does_not_raise_a_limit(self, default_account, empty_portfolio):
        """A high-confidence candidate gets the same size as a low-confidence
        one. Conviction belongs to the Signal Engine; size belongs here."""
        low = candidate()
        high = low.model_copy(update={"confidence": 0.99})
        sized = engine()
        assert (
            sized.authorize(low, default_account, empty_portfolio).lots
            == sized.authorize(high, default_account, empty_portfolio).lots
        )


class TestPortfolioStateHelpers:
    def test_exposure_sums_committed_max_loss(self, default_account):
        state = portfolio(
            acct=default_account,
            open_positions=[
                open_position(max_loss=Decimal(10000)),
                open_position(max_loss=Decimal(25000)),
            ],
        )
        assert state.total_exposure == Decimal(35000)
        assert state.exposure_for_underlying("NIFTY") == Decimal(35000)
        assert state.exposure_for_underlying("BANKNIFTY") == Decimal(0)

    def test_day_pnl_combines_realized_and_unrealized(self, default_account):
        state = portfolio(
            acct=default_account,
            daily_realized_pnl=Decimal(-5000),
            daily_unrealized_pnl=Decimal(2000),
        )
        assert state.day_pnl == Decimal(-3000)

    def test_counts_are_per_strategy_and_per_underlying(self, default_account):
        state = portfolio(
            acct=default_account,
            open_positions=[
                open_position(strategy=StrategyType.LONG_CALL),
                open_position(strategy=StrategyType.LONG_CALL, underlying="BANKNIFTY"),
            ],
        )
        assert state.count_for_strategy(StrategyType.LONG_CALL) == 2
        assert state.count_for_strategy(StrategyType.PUT_CREDIT_SPREAD) == 0
        assert state.count_for_underlying("NIFTY") == 1
        assert state.count_for_underlying("BANKNIFTY") == 1


class TestDecisionHelpers:
    def test_reject_produces_a_zeroed_decision(self):
        decision = RiskDecision.reject(RiskReasonCode.DAILY_LOSS_LIMIT_REACHED)
        assert decision.rejected
        assert decision.max_loss == 0
        assert decision.reason_codes == [RiskReasonCode.DAILY_LOSS_LIMIT_REACHED]

    @pytest.mark.parametrize("code", list(RiskReasonCode))
    def test_every_reason_code_round_trips(self, code: RiskReasonCode):
        assert RiskReasonCode(code.value) is code
