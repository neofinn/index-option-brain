"""Signal Engine behaviour (spec §11).

A signal is not an order, and must not come from single-indicator logic.
Three gates enforce that — scenario separation, cross-domain agreement, and
an absolute score floor — and each is tested here in isolation as well as
end to end.
"""

from __future__ import annotations

import pytest

from index_option_brain.brain.config import SignalEngineConfig
from index_option_brain.brain.pipeline import QuantitativeBrain
from index_option_brain.brain.signal_brain import DeterministicSignalEngine
from index_option_brain.contracts.analysis import AnalysisBundle, RegimeState
from index_option_brain.contracts.enums import (
    Direction,
    MarketRegimeType,
    ScenarioKind,
)
from index_option_brain.contracts.market_state import MarketState
from index_option_brain.contracts.scenario import Scenario
from tests.brain.test_regime_engine import (
    constituent_analysis,
    index_analysis,
    options_analysis,
    volatility_analysis,
)

engine = DeterministicSignalEngine()


def scenario(
    kind: ScenarioKind,
    direction: Direction,
    score: float,
    *,
    scenario_id: str = "s1",
    name: str = "test scenario",
) -> Scenario:
    return Scenario(
        scenario_id=scenario_id,
        kind=kind,
        name=name,
        direction=direction,
        score=score,
        confidence=0.8,
        supporting_evidence=["synthetic"],
    )


def state_with(
    base: MarketState,
    *,
    index_score: float = 0.8,
    breadth: float = 0.8,
    oi: float = 0.4,
    liquidity: float = 0.9,
    regime: MarketRegimeType = MarketRegimeType.TREND_UP,
) -> MarketState:
    """A state whose analysis is set explicitly, so one domain can be varied
    at a time without the simulator moving everything at once."""
    bundle = AnalysisBundle(
        index=index_analysis(
            trend_score=index_score, structure_score=index_score, momentum_score=index_score
        ),
        constituents=constituent_analysis(breadth_score=breadth, participation_score=0.9),
        options=options_analysis(oi_structure_score=oi, liquidity_score=liquidity),
        volatility=volatility_analysis(),
    )
    return base.with_analysis(bundle).with_regime(
        RegimeState(regime=regime, confidence=0.8)
    )


class TestDirectionalSignals:
    def test_aligned_domains_with_clear_separation_produce_a_signal(
        self, uptrend_state: MarketState
    ):
        state = state_with(uptrend_state)
        scenarios = [
            scenario(ScenarioKind.BULLISH_CONTINUATION, Direction.BULLISH, 0.85, scenario_id="a"),
            scenario(ScenarioKind.RANGE, Direction.NEUTRAL, 0.2, scenario_id="b"),
        ]
        signal = engine.evaluate(state, scenarios)
        assert signal.direction is Direction.BULLISH
        assert signal.score > 0.5
        assert signal.selected_scenario_id == "a"

    def test_the_signal_records_its_reasoning(self, uptrend_state: MarketState):
        state = state_with(uptrend_state)
        signal = engine.evaluate(
            state,
            [scenario(ScenarioKind.BULLISH_CONTINUATION, Direction.BULLISH, 0.85)],
        )
        assert signal.evidence
        assert any("agreement" in item for item in signal.evidence)


class TestSeparationGate:
    def test_an_evenly_matched_opposing_case_withholds_direction(
        self, uptrend_state: MarketState
    ):
        """Two futures pointing opposite ways and scoring alike means the
        evidence does not distinguish them."""
        state = state_with(uptrend_state)
        scenarios = [
            scenario(
                ScenarioKind.BULLISH_CONTINUATION, Direction.BULLISH, 0.62, scenario_id="a"
            ),
            scenario(
                ScenarioKind.BEARISH_CONTINUATION, Direction.BEARISH, 0.60, scenario_id="b"
            ),
        ]
        signal = engine.evaluate(state, scenarios)
        assert signal.direction is Direction.NEUTRAL
        assert any("separation" in item.lower() for item in signal.contradictions)

    def test_separation_is_measured_against_the_best_opposing_case(
        self, uptrend_state: MarketState
    ):
        """A second bullish scenario scoring alongside the leader is
        corroboration, not competition."""
        state = state_with(uptrend_state)
        scenarios = [
            scenario(
                ScenarioKind.BULLISH_CONTINUATION, Direction.BULLISH, 0.80, scenario_id="a"
            ),
            scenario(ScenarioKind.EXPANSION, Direction.BULLISH, 0.79, scenario_id="b"),
            scenario(ScenarioKind.RANGE, Direction.NEUTRAL, 0.10, scenario_id="c"),
        ]
        signal = engine.evaluate(state, scenarios)
        assert signal.direction is Direction.BULLISH


class TestAgreementGate:
    def test_breadth_contradicting_the_index_withholds_direction(
        self, uptrend_state: MarketState
    ):
        state = state_with(uptrend_state, index_score=0.9, breadth=-0.9, oi=-0.5)
        signal = engine.evaluate(
            state, [scenario(ScenarioKind.BULLISH_CONTINUATION, Direction.BULLISH, 0.8)]
        )
        assert signal.direction is Direction.NEUTRAL
        assert any("Breadth opposes" in item for item in signal.contradictions)

    def test_options_positioning_alone_cannot_carry_a_signal(
        self, uptrend_state: MarketState
    ):
        """Spec §7: OI is never a standalone BUY/SELL signal. With index and
        breadth flat, bullish positioning must not be enough."""
        state = state_with(uptrend_state, index_score=0.0, breadth=0.0, oi=1.0)
        signal = engine.evaluate(
            state, [scenario(ScenarioKind.BULLISH_CONTINUATION, Direction.BULLISH, 0.8)]
        )
        assert signal.direction is Direction.NEUTRAL

    def test_illiquid_chains_are_recorded_as_a_contradiction(
        self, uptrend_state: MarketState
    ):
        state = state_with(uptrend_state, liquidity=0.1)
        signal = engine.evaluate(
            state, [scenario(ScenarioKind.BULLISH_CONTINUATION, Direction.BULLISH, 0.85)]
        )
        assert any("liquidity" in item.lower() for item in signal.contradictions)


class TestNonDirectionalLeaders:
    def test_a_no_trade_leader_produces_a_neutral_signal(self, uptrend_state: MarketState):
        state = state_with(uptrend_state)
        signal = engine.evaluate(
            state, [scenario(ScenarioKind.NO_TRADE, Direction.NEUTRAL, 0.9)]
        )
        assert signal.direction is Direction.NEUTRAL
        assert signal.score == 0.0

    def test_a_range_leader_produces_a_neutral_signal(self, uptrend_state: MarketState):
        state = state_with(uptrend_state)
        signal = engine.evaluate(
            state, [scenario(ScenarioKind.RANGE, Direction.NEUTRAL, 0.9)]
        )
        assert signal.direction is Direction.NEUTRAL
        assert any("does not imply a direction" in item for item in signal.evidence)

    def test_no_scenarios_at_all_is_handled(self, uptrend_state: MarketState):
        signal = engine.evaluate(uptrend_state, [])
        assert signal.direction is Direction.NEUTRAL
        assert signal.contradictions


class TestRegimeInteraction:
    def test_an_opposing_regime_discounts_conviction(self, uptrend_state: MarketState):
        supportive = engine.evaluate(
            state_with(uptrend_state, regime=MarketRegimeType.TREND_UP),
            [scenario(ScenarioKind.BULLISH_CONTINUATION, Direction.BULLISH, 0.9)],
        )
        opposed = engine.evaluate(
            state_with(uptrend_state, regime=MarketRegimeType.TREND_DOWN),
            [scenario(ScenarioKind.BULLISH_CONTINUATION, Direction.BULLISH, 0.9)],
        )
        assert opposed.score < supportive.score

    def test_an_uncertain_regime_discounts_conviction(self, uptrend_state: MarketState):
        certain = engine.evaluate(
            state_with(uptrend_state, regime=MarketRegimeType.TREND_UP),
            [scenario(ScenarioKind.BULLISH_CONTINUATION, Direction.BULLISH, 0.9)],
        )
        uncertain = engine.evaluate(
            state_with(uptrend_state, regime=MarketRegimeType.UNCERTAIN),
            [scenario(ScenarioKind.BULLISH_CONTINUATION, Direction.BULLISH, 0.9)],
        )
        assert uncertain.score < certain.score


class TestConfirmationFlag:
    def test_a_marginal_signal_requires_confirmation(self, uptrend_state: MarketState):
        lenient = DeterministicSignalEngine(
            SignalEngineConfig(min_score=0.1, confirmation_score=0.95)
        )
        signal = lenient.evaluate(
            state_with(uptrend_state),
            [scenario(ScenarioKind.BULLISH_CONTINUATION, Direction.BULLISH, 0.5)],
        )
        assert signal.direction is Direction.BULLISH
        assert signal.confirmation_required


class TestEndToEnd:
    @pytest.mark.parametrize(
        "fixture_name,expected",
        [
            ("uptrend_state", Direction.BULLISH),
            ("downtrend_state", Direction.BEARISH),
            ("range_state", Direction.NEUTRAL),
            ("narrow_rally_state", Direction.NEUTRAL),
        ],
    )
    def test_simulated_markets_produce_the_expected_direction(
        self, request, fixture_name: str, expected: Direction
    ):
        state: MarketState = request.getfixturevalue(fixture_name)
        result = QuantitativeBrain().run(state)
        assert result.signal.direction is expected
