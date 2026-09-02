"""Volatility Engine behaviour (spec §8).

The distinction under test is between *level* (where IV sits in its own
history) and *richness* (IV against realized). They are separate fields
because they answer separate questions, and conflating them is how premium
gets sold into a market that is genuinely moving.
"""

from __future__ import annotations

from index_option_brain.brain.volatility_brain import DeterministicVolatilityEngine
from index_option_brain.contracts.enums import IvRegime
from index_option_brain.contracts.market_state import MarketState

engine = DeterministicVolatilityEngine()


class TestRegimeClassification:
    def test_iv_at_the_top_of_its_history_reads_high(self, state_builder):
        state = state_builder(base_iv=20.0, iv_history=[10.0 + i * 0.1 for i in range(40)])
        analysis = engine.analyze(state)
        assert analysis.regime is IvRegime.HIGH
        assert analysis.iv_percentile == 1.0

    def test_iv_at_the_bottom_of_its_history_reads_low(self, state_builder):
        state = state_builder(base_iv=8.0, iv_history=[15.0 + i * 0.1 for i in range(40)])
        analysis = engine.analyze(state)
        assert analysis.regime is IvRegime.LOW
        # The current print is part of its own history, so the floor is 1/n.
        assert analysis.iv_percentile is not None
        assert analysis.iv_percentile < 0.05

    def test_too_little_history_defaults_to_normal_rather_than_ranking(
        self, uptrend_state: MarketState
    ):
        """One print is not a distribution. Ranking against it would report
        the first observation of a new series as a volatility extreme."""
        thin = uptrend_state.volatility_state.model_copy(update={"atm_iv_history": [14.0]})
        analysis = engine.analyze(uptrend_state.model_copy(update={"volatility_state": thin}))
        assert analysis.regime is IvRegime.NORMAL
        assert analysis.iv_percentile is None
        assert any("too few to rank" in item for item in analysis.evidence)


class TestRichness:
    def test_iv_above_realized_reads_rich(self, state_builder):
        state = state_builder(daily_volatility_pct=0.3, base_iv=22.0, mean_reversion=0.5)
        analysis = engine.analyze(state)
        assert analysis.iv_rv_ratio is not None and analysis.iv_rv_ratio > 1
        assert analysis.iv_score > 0

    def test_iv_below_realized_reads_cheap(self, cheap_volatility_state: MarketState):
        analysis = engine.analyze(cheap_volatility_state)
        assert analysis.iv_rv_ratio is not None and analysis.iv_rv_ratio < 1
        assert analysis.iv_score < 0

    def test_level_and_richness_are_independent(self, cheap_volatility_state: MarketState):
        """High IV is not the same as expensive IV: here IV sits low in its
        own history while realized volatility is higher still."""
        analysis = engine.analyze(cheap_volatility_state)
        assert analysis.regime is IvRegime.LOW
        assert analysis.iv_score < 0
        assert analysis.realized_volatility is not None
        assert analysis.atm_iv is not None
        assert analysis.atm_iv < analysis.realized_volatility


class TestExpectedMove:
    def test_expected_move_is_positive_and_scales_with_time(self, state_builder):
        near = engine.analyze(state_builder(expiry_index=0))
        far = engine.analyze(state_builder(expiry_index=3))
        assert near.expected_move > 0
        assert far.expected_move > near.expected_move

    def test_expected_move_scales_with_implied_volatility(self, state_builder):
        calm = engine.analyze(state_builder(base_iv=10.0))
        stormy = engine.analyze(state_builder(base_iv=30.0))
        assert stormy.expected_move > calm.expected_move

    def test_no_implied_volatility_yields_no_expected_move(self, uptrend_state: MarketState):
        stripped = uptrend_state.volatility_state.model_copy(update={"atm_iv": None})
        blank_chain = uptrend_state.options_state.model_copy(update={"chain": []})
        analysis = engine.analyze(
            uptrend_state.model_copy(
                update={"volatility_state": stripped, "options_state": blank_chain}
            )
        )
        assert analysis.expected_move == 0
        assert analysis.confidence == 0.0


class TestFallbacksAndConfidence:
    def test_atm_iv_falls_back_to_the_chain(self, uptrend_state: MarketState):
        """If the data layer didn't supply an ATM IV, take it from the chain
        rather than assuming a level."""
        stripped = uptrend_state.volatility_state.model_copy(update={"atm_iv": None})
        analysis = engine.analyze(uptrend_state.model_copy(update={"volatility_state": stripped}))
        assert analysis.atm_iv is not None
        assert analysis.atm_iv > 0

    def test_confidence_rises_with_history_length(self, state_builder):
        thin = engine.analyze(state_builder(iv_history=[14.0, 14.1]))
        thick = engine.analyze(state_builder(iv_history=[14.0 + i * 0.05 for i in range(40)]))
        assert thick.confidence > thin.confidence

    def test_expansion_is_reported_with_evidence(self, uptrend_state: MarketState):
        analysis = engine.analyze(uptrend_state)
        assert -1.0 <= analysis.expansion_score <= 1.0
        assert analysis.evidence
