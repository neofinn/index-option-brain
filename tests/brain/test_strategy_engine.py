"""Strategy Engine behaviour (spec §12).

The central behaviour is that the *expression* of a view depends on
volatility, not just direction: the same bullish signal should produce a
credit structure when premium is rich and a debit structure when it is
cheap. The other invariant is that NO_TRADE is always present and wins
whenever nothing else earns its place.
"""

from __future__ import annotations

from index_option_brain.brain.pipeline import QuantitativeBrain
from index_option_brain.brain.strategy_brain import DeterministicStrategyEngine
from index_option_brain.contracts.enums import Direction, StrategyType
from index_option_brain.contracts.market_state import MarketState
from index_option_brain.contracts.signal import Signal

engine = DeterministicStrategyEngine()

_CREDIT = {
    StrategyType.CALL_CREDIT_SPREAD,
    StrategyType.PUT_CREDIT_SPREAD,
    StrategyType.NEUTRAL_DEFINED_RISK,
}
_LONG_PREMIUM = {StrategyType.LONG_CALL, StrategyType.LONG_PUT}


def run(state: MarketState):
    return QuantitativeBrain().run(state)


def signal(direction: Direction, score: float = 0.8) -> Signal:
    return Signal(
        signal_id="sig",
        direction=direction,
        score=score,
        confidence=0.8,
        selected_scenario_id="scn",
        confirmation_required=False,
    )


class TestNoTradeAlwaysValid:
    def test_no_trade_is_always_among_the_candidates(
        self,
        uptrend_state: MarketState,
        downtrend_state: MarketState,
        range_state: MarketState,
        narrow_rally_state: MarketState,
    ):
        for state in (uptrend_state, downtrend_state, range_state, narrow_rally_state):
            result = run(state)
            kinds = {c.strategy for c in result.strategy_candidates}
            assert StrategyType.NO_TRADE in kinds

    def test_no_trade_wins_when_the_signal_is_neutral_and_nothing_fits(
        self, narrow_rally_state: MarketState
    ):
        result = run(narrow_rally_state)
        assert result.signal.direction is Direction.NEUTRAL
        assert result.selected_strategy is StrategyType.NO_TRADE
        assert not result.is_actionable

    def test_no_trade_outranks_a_structure_below_the_acceptance_floor(
        self, uptrend_state: MarketState
    ):
        """A weak candidate must never win by being the only candidate."""
        candidates = engine.select(run(uptrend_state).state, signal(Direction.BULLISH, 0.05))
        best = candidates[0]
        assert best.strategy is StrategyType.NO_TRADE

    def test_no_trade_carries_its_reasons(self, narrow_rally_state: MarketState):
        result = run(narrow_rally_state)
        no_trade = next(
            c for c in result.strategy_candidates if c.strategy is StrategyType.NO_TRADE
        )
        assert no_trade.rationale


class TestVolatilityAwareExpression:
    def test_rich_premium_prefers_collecting_it(self, uptrend_state: MarketState):
        result = run(uptrend_state)
        assert result.analysis.volatility.iv_score > 0.25
        assert result.signal.direction is Direction.BULLISH
        assert result.selected_strategy in _CREDIT

    def test_cheap_premium_prefers_paying_it(self, cheap_volatility_state: MarketState):
        result = run(cheap_volatility_state)
        assert result.analysis.volatility.iv_score < -0.25
        assert result.selected_strategy not in _CREDIT
        assert result.selected_strategy is not StrategyType.NO_TRADE

    def test_the_same_direction_produces_different_structures_by_volatility(
        self, uptrend_state: MarketState, cheap_volatility_state: MarketState
    ):
        """The point of the engine: direction alone does not determine the
        trade."""
        rich = run(uptrend_state)
        cheap = run(cheap_volatility_state)
        assert rich.signal.direction is cheap.signal.direction is Direction.BULLISH
        assert rich.selected_strategy is not cheap.selected_strategy

    def test_bearish_signals_mirror_the_bullish_logic(self, downtrend_state: MarketState):
        result = run(downtrend_state)
        assert result.signal.direction is Direction.BEARISH
        assert result.selected_strategy in {
            StrategyType.CALL_CREDIT_SPREAD,
            StrategyType.PUT_DEBIT_SPREAD,
            StrategyType.LONG_PUT,
        }


class TestExpiryConstraints:
    def test_long_premium_is_excluded_close_to_expiry(self, state_builder):
        """Buying premium into the last day is a theta trap, so directional
        views that close to expiry are only offered as spreads."""
        from tests.conftest import PINNED_EXPIRY_DAY

        state = state_builder(
            daily_drift_pct=0.35,
            intraday_drift_pct=2.0,
            breadth_bias=0.6,
            base_iv=9.0,
            daily_volatility_pct=1.6,
            iv_history=[9.0 + (i % 5) * 0.2 for i in range(40)],
            as_of=PINNED_EXPIRY_DAY,
        )
        result = run(state)
        assert (state.volatility_state.days_to_expiry or 0) < 2.0
        offered = {c.strategy for c in result.strategy_candidates}
        assert not (offered & _LONG_PREMIUM)

    def test_long_premium_is_available_with_time_left(
        self, cheap_volatility_state: MarketState
    ):
        result = run(cheap_volatility_state)
        assert (cheap_volatility_state.volatility_state.days_to_expiry or 0) > 2.0
        offered = {c.strategy for c in result.strategy_candidates}
        assert offered & _LONG_PREMIUM


class TestNeutralOpportunities:
    def test_a_range_with_rich_premium_offers_a_defined_risk_neutral_structure(
        self, range_state: MarketState
    ):
        """A neutral signal is the absence of a *directional* opportunity,
        not necessarily the absence of any."""
        result = run(range_state)
        assert result.signal.direction is Direction.NEUTRAL
        assert result.analysis.volatility.iv_score > 0.25
        assert result.selected_strategy is StrategyType.NEUTRAL_DEFINED_RISK

    def test_the_neutral_structure_is_a_credit_with_defined_risk(
        self, range_state: MarketState
    ):
        result = run(range_state)
        candidate = result.best_candidate
        assert candidate is not None
        assert candidate.is_credit
        assert candidate.max_loss > 0
        assert candidate.max_profit is not None
        assert len(candidate.legs) == 4


class TestBlockers:
    def test_an_illiquid_chain_blocks_every_structure(self, uptrend_state: MarketState):
        unquoted = [
            q.model_copy(update={"bid": None, "ask": None})
            for q in uptrend_state.options_state.chain
        ]
        options_state = uptrend_state.options_state.model_copy(update={"chain": unquoted})
        result = run(uptrend_state.model_copy(update={"options_state": options_state}))
        assert result.selected_strategy is StrategyType.NO_TRADE
        no_trade = result.strategy_candidates[0]
        assert "liquidity" in no_trade.rationale.lower()

    def test_an_incomplete_chain_blocks_options_entry(self, uptrend_state: MarketState):
        """Spec §29: incomplete option-chain data means no options entry."""
        from index_option_brain.contracts.enums import OptionType

        calls_only = [
            q
            for q in uptrend_state.options_state.chain
            if q.contract.option_type is OptionType.CE
        ]
        options_state = uptrend_state.options_state.model_copy(update={"chain": calls_only})
        result = run(uptrend_state.model_copy(update={"options_state": options_state}))
        assert result.selected_strategy is StrategyType.NO_TRADE

    def test_no_analysis_yields_no_trade(self, uptrend_state: MarketState):
        candidates = engine.select(
            uptrend_state.model_copy(update={"analysis": None}), signal(Direction.BULLISH)
        )
        assert candidates[0].strategy is StrategyType.NO_TRADE


class TestCandidateEconomics:
    def test_every_candidate_reports_max_loss_and_a_rationale(
        self, uptrend_state: MarketState
    ):
        result = run(uptrend_state)
        for candidate in result.strategy_candidates:
            assert candidate.max_loss >= 0
            assert candidate.rationale
            assert 0.0 <= candidate.score <= 1.0

    def test_candidates_are_returned_best_first(self, uptrend_state: MarketState):
        scores = [c.score for c in run(uptrend_state).strategy_candidates]
        assert scores == sorted(scores, reverse=True)

    def test_defined_risk_structures_bound_their_loss(self, uptrend_state: MarketState):
        result = run(uptrend_state)
        for candidate in result.strategy_candidates:
            if candidate.strategy in _CREDIT:
                assert candidate.max_loss > 0
                assert candidate.max_profit is not None

    def test_long_options_report_unbounded_upside(self, cheap_volatility_state: MarketState):
        result = run(cheap_volatility_state)
        for candidate in result.strategy_candidates:
            if candidate.strategy in _LONG_PREMIUM:
                assert candidate.max_profit is None
                assert candidate.breakeven
