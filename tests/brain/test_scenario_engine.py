"""Scenario Engine behaviour (spec §10).

Two properties matter most: NO_TRADE is always generated and scores highest
when it should, and competing scenarios are constructed honestly — each
carrying the evidence against it, since the Signal Engine's separation check
is worthless if the rivals are straw men.
"""

from __future__ import annotations

from index_option_brain.brain.pipeline import QuantitativeBrain
from index_option_brain.brain.scenario_brain import DeterministicScenarioEngine
from index_option_brain.contracts.analysis import RegimeState
from index_option_brain.contracts.enums import Direction, MarketRegimeType, ScenarioKind
from index_option_brain.contracts.market_state import MarketState

engine = DeterministicScenarioEngine()


def scenarios_for(state: MarketState):
    """Run the analysis stages the Scenario Engine depends on, then generate."""
    brain = QuantitativeBrain()
    result = brain.run(state)
    return result.scenarios


def by_kind(scenarios, kind: ScenarioKind):
    return next((s for s in scenarios if s.kind is kind), None)


class TestNoTradeIsAlwaysAvailable:
    def test_no_trade_is_generated_for_every_market(
        self,
        uptrend_state: MarketState,
        downtrend_state: MarketState,
        range_state: MarketState,
        narrow_rally_state: MarketState,
    ):
        for state in (uptrend_state, downtrend_state, range_state, narrow_rally_state):
            assert by_kind(scenarios_for(state), ScenarioKind.NO_TRADE) is not None

    def test_no_trade_scores_higher_when_conviction_is_thinner(
        self, uptrend_state: MarketState, narrow_rally_state: MarketState
    ):
        """NO_TRADE strengthens as the case for anything else weakens: a
        heavyweight-driven rally should make standing aside more attractive
        than a broad, participated one does."""
        broad = by_kind(scenarios_for(uptrend_state), ScenarioKind.NO_TRADE)
        narrow = by_kind(scenarios_for(narrow_rally_state), ScenarioKind.NO_TRADE)
        assert broad is not None and narrow is not None
        assert narrow.score > broad.score

    def test_a_contested_rally_weakens_its_own_directional_case(
        self, uptrend_state: MarketState, narrow_rally_state: MarketState
    ):
        broad = by_kind(scenarios_for(uptrend_state), ScenarioKind.BULLISH_CONTINUATION)
        narrow = by_kind(
            scenarios_for(narrow_rally_state), ScenarioKind.BULLISH_CONTINUATION
        )
        assert broad is not None and narrow is not None
        assert narrow.score < broad.score

    def test_no_trade_loses_to_a_well_supported_case(self, uptrend_state: MarketState):
        scenarios = scenarios_for(uptrend_state)
        no_trade = by_kind(scenarios, ScenarioKind.NO_TRADE)
        bullish = by_kind(scenarios, ScenarioKind.BULLISH_CONTINUATION)
        assert no_trade is not None and bullish is not None
        assert bullish.score > no_trade.score

    def test_missing_analysis_yields_only_no_trade(self, uptrend_state: MarketState):
        bare = uptrend_state.model_copy(update={"analysis": None})
        scenarios = engine.generate(
            bare, RegimeState(regime=MarketRegimeType.UNCERTAIN, confidence=0.0)
        )
        assert len(scenarios) == 1
        assert scenarios[0].kind is ScenarioKind.NO_TRADE


class TestDirectionalScenarios:
    def test_an_uptrend_produces_a_leading_bullish_scenario(
        self, uptrend_state: MarketState
    ):
        scenarios = scenarios_for(uptrend_state)
        leader = max(scenarios, key=lambda s: s.score)
        assert leader.kind is ScenarioKind.BULLISH_CONTINUATION
        assert leader.direction is Direction.BULLISH

    def test_a_downtrend_produces_a_leading_bearish_scenario(
        self, downtrend_state: MarketState
    ):
        leader = max(scenarios_for(downtrend_state), key=lambda s: s.score)
        assert leader.kind is ScenarioKind.BEARISH_CONTINUATION
        assert leader.direction is Direction.BEARISH

    def test_opposing_continuations_are_not_generated_together(
        self, uptrend_state: MarketState
    ):
        """Bullish and bearish continuation are mutually exclusive readings of
        the same composite score — only one can be argued from the data."""
        scenarios = scenarios_for(uptrend_state)
        assert by_kind(scenarios, ScenarioKind.BEARISH_CONTINUATION) is None


class TestRangeAndVolatilityScenarios:
    def test_a_range_market_produces_a_leading_range_scenario(
        self, range_state: MarketState
    ):
        scenarios = scenarios_for(range_state)
        range_scenario = by_kind(scenarios, ScenarioKind.RANGE)
        assert range_scenario is not None
        assert range_scenario.direction is Direction.NEUTRAL
        assert range_scenario.score > 0.3

    def test_a_range_scenario_is_neutral_not_untradeable(self, range_state: MarketState):
        """RANGE implies no *direction*, which is different from implying no
        opportunity — the Strategy Engine may still express it."""
        range_scenario = by_kind(scenarios_for(range_state), ScenarioKind.RANGE)
        assert range_scenario is not None
        assert range_scenario.is_tradeable


class TestEvidenceQuality:
    def test_every_scenario_carries_supporting_evidence(self, uptrend_state: MarketState):
        for scenario in scenarios_for(uptrend_state):
            assert scenario.supporting_evidence

    def test_a_contested_scenario_records_what_argues_against_it(
        self, narrow_rally_state: MarketState
    ):
        """Honest rivals are the precondition for the Signal Engine's
        separation check meaning anything."""
        bullish = by_kind(
            scenarios_for(narrow_rally_state), ScenarioKind.BULLISH_CONTINUATION
        )
        assert bullish is not None
        assert bullish.contradictory_evidence

    def test_directional_scenarios_state_confirmation_and_invalidation(
        self, uptrend_state: MarketState
    ):
        leader = max(scenarios_for(uptrend_state), key=lambda s: s.score)
        assert leader.confirmation_conditions
        assert leader.invalidation_conditions

    def test_scenario_ids_are_unique(self, uptrend_state: MarketState):
        scenarios = scenarios_for(uptrend_state)
        ids = [s.scenario_id for s in scenarios]
        assert len(ids) == len(set(ids))


class TestOutputShape:
    def test_scenarios_are_returned_best_first(self, uptrend_state: MarketState):
        scores = [s.score for s in scenarios_for(uptrend_state)]
        assert scores == sorted(scores, reverse=True)

    def test_scores_and_confidences_are_bounded(self, uptrend_state: MarketState):
        for scenario in scenarios_for(uptrend_state):
            assert 0.0 <= scenario.score <= 1.0
            assert 0.0 <= scenario.confidence <= 1.0
