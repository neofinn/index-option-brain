"""Strike Engine behaviour (spec §13).

Hard filters run before scoring, so an untradeable contract is never
proposed rather than being proposed and then rejected at the gate. Ranking
then balances delta fit, liquidity, and structural quality — any one of
those alone picks predictably bad strikes.
"""

from __future__ import annotations

from index_option_brain.brain.config import StrikeEngineConfig
from index_option_brain.brain.pipeline import QuantitativeBrain
from index_option_brain.brain.strike_brain import DeterministicStrikeEngine
from index_option_brain.contracts.enums import OptionType, OrderSide, StrategyType
from index_option_brain.contracts.market_state import MarketState

engine = DeterministicStrikeEngine()


def analysed(state: MarketState) -> MarketState:
    """A state carrying its analysis, as the Strike Engine expects."""
    return QuantitativeBrain().run(state).state


class TestRanking:
    def test_candidates_are_returned_best_first(self, uptrend_state: MarketState):
        state = analysed(uptrend_state)
        candidates = engine.rank(
            StrategyType.PUT_CREDIT_SPREAD, state.options_state.chain, state
        )
        assert candidates
        scores = [c.score for c in candidates]
        assert scores == sorted(scores, reverse=True)

    def test_the_candidate_count_is_capped(self, uptrend_state: MarketState):
        state = analysed(uptrend_state)
        capped = DeterministicStrikeEngine(StrikeEngineConfig(max_candidates=2))
        assert (
            len(capped.rank(StrategyType.PUT_CREDIT_SPREAD, state.options_state.chain, state))
            <= 2
        )

    def test_no_trade_produces_no_candidates(self, uptrend_state: MarketState):
        state = analysed(uptrend_state)
        assert engine.rank(StrategyType.NO_TRADE, state.options_state.chain, state) == []

    def test_an_empty_chain_produces_no_candidates(self, uptrend_state: MarketState):
        assert engine.rank(StrategyType.LONG_CALL, [], analysed(uptrend_state)) == []

    def test_every_candidate_carries_a_readable_rationale(self, uptrend_state: MarketState):
        state = analysed(uptrend_state)
        for candidate in engine.rank(
            StrategyType.CALL_DEBIT_SPREAD, state.options_state.chain, state
        ):
            assert candidate.rationale
            assert 0.0 <= candidate.score <= 1.0


class TestHardFilters:
    def test_wide_spreads_are_filtered_out_entirely(self, uptrend_state: MarketState):
        """Not merely down-ranked: spec §16 would reject them at the gate, so
        proposing them wastes a cycle and misleads the review trail."""
        state = analysed(uptrend_state)
        candidates = engine.rank(
            StrategyType.LONG_CALL, state.options_state.chain, state
        )
        assert candidates
        limit = StrikeEngineConfig().max_relative_spread
        assert all(c.worst_relative_spread <= limit for c in candidates)

    def test_thin_open_interest_is_filtered_out(self, uptrend_state: MarketState):
        state = analysed(uptrend_state)
        strict = DeterministicStrikeEngine(StrikeEngineConfig(min_open_interest=10_000_000))
        assert strict.rank(StrategyType.LONG_CALL, state.options_state.chain, state) == []

    def test_one_sided_quotes_are_filtered_out(self, uptrend_state: MarketState):
        state = analysed(uptrend_state)
        one_sided = [q.model_copy(update={"bid": None}) for q in state.options_state.chain]
        assert engine.rank(StrategyType.LONG_CALL, one_sided, state) == []

    def test_every_surviving_candidate_has_a_positive_max_loss(
        self, uptrend_state: MarketState
    ):
        state = analysed(uptrend_state)
        for strategy in (
            StrategyType.LONG_CALL,
            StrategyType.CALL_DEBIT_SPREAD,
            StrategyType.PUT_CREDIT_SPREAD,
        ):
            for candidate in engine.rank(strategy, state.options_state.chain, state):
                assert candidate.max_loss > 0


class TestDeltaFit:
    def test_directional_structures_prefer_meaningful_exposure(
        self, cheap_volatility_state: MarketState
    ):
        state = analysed(cheap_volatility_state)
        candidates = engine.rank(StrategyType.LONG_CALL, state.options_state.chain, state)
        assert candidates
        best = candidates[0]
        delta = abs(float(best.legs[0].delta or 0))
        target = StrikeEngineConfig().directional_target_delta
        tolerance = StrikeEngineConfig().delta_tolerance
        assert abs(delta - target) <= tolerance

    def test_credit_structures_prefer_strikes_unlikely_to_be_reached(
        self, uptrend_state: MarketState
    ):
        """A premium seller wants a lower-delta short strike than a
        directional buyer wants for their long."""
        state = analysed(uptrend_state)
        credit = engine.rank(
            StrategyType.PUT_CREDIT_SPREAD, state.options_state.chain, state
        )
        debit = engine.rank(
            StrategyType.CALL_DEBIT_SPREAD, state.options_state.chain, state
        )
        assert credit and debit

        short_leg = next(leg for leg in credit[0].legs if leg.side is OrderSide.SELL)
        long_leg = next(leg for leg in debit[0].legs if leg.side is OrderSide.BUY)
        assert abs(float(short_leg.delta or 0)) < abs(float(long_leg.delta or 0))


class TestStructureShape:
    def test_a_long_call_is_a_single_bought_call(self, cheap_volatility_state: MarketState):
        state = analysed(cheap_volatility_state)
        candidate = engine.rank(StrategyType.LONG_CALL, state.options_state.chain, state)[0]
        assert len(candidate.legs) == 1
        assert candidate.legs[0].side is OrderSide.BUY
        assert candidate.legs[0].contract.option_type is OptionType.CE
        assert not candidate.is_credit

    def test_a_put_credit_spread_sells_the_higher_strike(self, uptrend_state: MarketState):
        state = analysed(uptrend_state)
        candidate = engine.rank(
            StrategyType.PUT_CREDIT_SPREAD, state.options_state.chain, state
        )[0]
        short_leg = next(leg for leg in candidate.legs if leg.side is OrderSide.SELL)
        long_leg = next(leg for leg in candidate.legs if leg.side is OrderSide.BUY)
        assert short_leg.contract.strike > long_leg.contract.strike
        assert candidate.is_credit

    def test_credit_spreads_are_collateralized_by_their_max_loss(
        self, uptrend_state: MarketState
    ):
        state = analysed(uptrend_state)
        candidate = engine.rank(
            StrategyType.PUT_CREDIT_SPREAD, state.options_state.chain, state
        )[0]
        assert candidate.capital_required == candidate.max_loss

    def test_a_condor_has_four_legs_and_two_breakevens(self, range_state: MarketState):
        state = analysed(range_state)
        candidates = engine.rank(
            StrategyType.NEUTRAL_DEFINED_RISK, state.options_state.chain, state
        )
        assert candidates
        assert len(candidates[0].legs) == 4
        assert len(candidates[0].breakeven) == 2


class TestWallAwareness:
    def test_buying_into_a_wall_is_penalized_and_noted(self, cheap_volatility_state: MarketState):
        """Paying for a move that positioning is leaning against."""
        state = analysed(cheap_volatility_state)
        call_walls = {float(w) for w in state.analysis.options.call_walls}
        candidates = engine.rank(StrategyType.LONG_CALL, state.options_state.chain, state)
        at_wall = [
            c
            for c in candidates
            if float(c.legs[0].contract.strike) in call_walls
        ]
        for candidate in at_wall:
            assert "call wall" in candidate.rationale

    def test_selling_at_a_wall_is_noted_as_favourable(self, uptrend_state: MarketState):
        state = analysed(uptrend_state)
        put_walls = {float(w) for w in state.analysis.options.put_walls}
        candidates = engine.rank(
            StrategyType.PUT_CREDIT_SPREAD, state.options_state.chain, state
        )
        for candidate in candidates:
            short_leg = next(leg for leg in candidate.legs if leg.side is OrderSide.SELL)
            if float(short_leg.contract.strike) in put_walls:
                assert "put wall" in candidate.rationale
                break


class TestPipelineIntegration:
    def test_the_pipeline_stands_down_if_no_contract_survives_filtering(
        self, uptrend_state: MarketState
    ):
        """A structure that cannot actually be expressed is not a trade, so
        the selected strategy must fall back to NO_TRADE."""
        brain = QuantitativeBrain(
            strike_engine=DeterministicStrikeEngine(
                StrikeEngineConfig(min_open_interest=10_000_000)
            )
        )
        result = brain.run(uptrend_state)
        assert result.strike_candidates == []
        assert result.selected_strategy is StrategyType.NO_TRADE
        assert not result.is_actionable
