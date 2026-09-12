"""Index Brain behaviour (spec §5)."""

from __future__ import annotations

from index_option_brain.brain.index_brain import DeterministicIndexBrain
from index_option_brain.contracts.enums import BreakoutState, Direction, VwapRelationship
from index_option_brain.contracts.market_state import MarketState

brain = DeterministicIndexBrain()


class TestDirection:
    def test_a_participated_uptrend_reads_bullish(self, uptrend_state: MarketState):
        analysis = brain.analyze(uptrend_state)
        assert analysis.direction is Direction.BULLISH
        assert analysis.trend_score > 0
        assert analysis.structure_score > 0
        assert analysis.momentum_score > 0
        assert analysis.confidence > 0.5

    def test_a_downtrend_reads_bearish(self, downtrend_state: MarketState):
        analysis = brain.analyze(downtrend_state)
        assert analysis.direction is Direction.BEARISH
        assert analysis.composite_score < 0

    def test_a_range_reads_neutral(self, range_state: MarketState):
        """A flat market must not be forced into a direction — NEUTRAL is a
        legitimate read, not a failure to decide."""
        analysis = brain.analyze(range_state)
        assert analysis.direction is Direction.NEUTRAL
        assert abs(analysis.composite_score) < 0.25


class TestScoreBounds:
    def test_all_scores_stay_within_their_declared_ranges(
        self, uptrend_state: MarketState, downtrend_state: MarketState, range_state: MarketState
    ):
        for state in (uptrend_state, downtrend_state, range_state):
            analysis = brain.analyze(state)
            assert -1.0 <= analysis.trend_score <= 1.0
            assert -1.0 <= analysis.structure_score <= 1.0
            assert -1.0 <= analysis.momentum_score <= 1.0
            assert 0.0 <= analysis.volatility_score <= 1.0
            assert 0.0 <= analysis.confidence <= 1.0


class TestStructuralDetail:
    def test_levels_bracket_the_spot(self, uptrend_state: MarketState):
        analysis = brain.analyze(uptrend_state)
        spot = uptrend_state.spot
        assert all(level < spot for level in analysis.support_levels)
        assert all(level > spot for level in analysis.resistance_levels)

    def test_supports_are_ordered_nearest_first(self, downtrend_state: MarketState):
        analysis = brain.analyze(downtrend_state)
        assert analysis.support_levels == sorted(analysis.support_levels, reverse=True)
        assert analysis.resistance_levels == sorted(analysis.resistance_levels)

    def test_atr_is_reported_and_positive(self, uptrend_state: MarketState):
        analysis = brain.analyze(uptrend_state)
        assert analysis.atr is not None
        assert analysis.atr > 0

    def test_a_strong_advance_registers_a_breakout(self, uptrend_state: MarketState):
        analysis = brain.analyze(uptrend_state)
        assert analysis.breakout_state in (
            BreakoutState.BREAKOUT,
            BreakoutState.FAILED_BREAKOUT,
        )

    def test_a_range_registers_no_break(self, range_state: MarketState):
        assert brain.analyze(range_state).breakout_state is BreakoutState.NONE

    def test_vwap_relationship_agrees_with_its_distance(self, uptrend_state: MarketState):
        analysis = brain.analyze(uptrend_state)
        if analysis.vwap_relationship is VwapRelationship.ABOVE:
            assert analysis.vwap_distance_atr > 0
        elif analysis.vwap_relationship is VwapRelationship.BELOW:
            assert analysis.vwap_distance_atr < 0

    def test_range_positions_are_normalized(self, uptrend_state: MarketState):
        analysis = brain.analyze(uptrend_state)
        for position in (analysis.day_range_position, analysis.opening_range_position):
            if position is not None:
                assert 0.0 <= position <= 1.0


class TestEvidence:
    def test_evidence_is_produced_and_human_readable(self, uptrend_state: MarketState):
        """Spec §31 requires a trade be reconstructable; a score with no
        stated reasoning cannot be reviewed after the fact."""
        analysis = brain.analyze(uptrend_state)
        assert len(analysis.evidence) >= 3
        assert all(isinstance(item, str) and item for item in analysis.evidence)

    def test_a_direction_comes_with_invalidation_conditions(self, uptrend_state: MarketState):
        analysis = brain.analyze(uptrend_state)
        assert analysis.direction is Direction.BULLISH
        assert analysis.invalidations, "a directional read must say what would break it"


class TestDegradedInput:
    def test_a_bare_snapshot_yields_low_confidence_not_an_error(
        self, uptrend_state: MarketState
    ):
        """With no history there is nothing to measure trend or structure
        against, so the brain must report that rather than guessing."""
        stripped = uptrend_state.index_state.model_copy(
            update={"daily_bars": [], "intraday_bars": [], "opening_range": None}
        )
        analysis = brain.analyze(uptrend_state.model_copy(update={"index_state": stripped}))
        assert analysis.confidence < 0.3
        assert analysis.atr is None
        assert analysis.breakout_state is BreakoutState.NONE

    def test_confidence_rises_with_available_history(self, state_builder):
        short_history = state_builder(
            daily_drift_pct=0.35, intraday_drift_pct=2.0, breadth_bias=0.6
        )
        trimmed = short_history.index_state.model_copy(
            update={"daily_bars": short_history.index_state.daily_bars[-10:]}
        )
        thin = brain.analyze(short_history.model_copy(update={"index_state": trimmed}))
        full = brain.analyze(short_history)
        assert full.confidence > thin.confidence
