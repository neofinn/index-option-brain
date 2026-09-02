"""End-to-end pipeline behaviour (spec §33).

The pipeline's most important property is where it *stops*: at ranked
contracts, with no TradeDecision and no risk approval. Constructing a
TradeDecision here would require inventing a RiskDecision, and a fabricated
risk approval is exactly the placeholder-as-production that spec §36
prohibits.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from index_option_brain.brain.pipeline import BrainCycleResult, QuantitativeBrain
from index_option_brain.contracts.enums import Direction, MarketRegimeType, StrategyType
from index_option_brain.contracts.market_state import MarketState

brain = QuantitativeBrain()


class TestStageWiring:
    def test_every_stage_produces_output(self, uptrend_state: MarketState):
        result = brain.run(uptrend_state)
        assert result.analysis.index is not None
        assert result.analysis.constituents is not None
        assert result.analysis.options is not None
        assert result.analysis.volatility is not None
        assert result.regime is not None
        assert result.scenarios
        assert result.signal is not None
        assert result.strategy_candidates

    def test_each_stage_output_is_folded_back_into_state(self, uptrend_state: MarketState):
        """Downstream stages read state, not positional arguments, so the
        state carried out of a cycle is the full record of that cycle."""
        result = brain.run(uptrend_state)
        assert result.state.analysis == result.analysis
        assert result.state.market_regime == result.regime
        assert result.state.active_scenarios == result.scenarios
        assert result.state.active_signals == [result.signal]

    def test_the_input_state_is_not_mutated(self, uptrend_state: MarketState):
        """MarketState is frozen and advanced by copy, so a stage can never
        mutate the snapshot another stage is reading."""
        assert uptrend_state.analysis is None
        brain.run(uptrend_state)
        assert uptrend_state.analysis is None
        assert uptrend_state.market_regime is None
        assert uptrend_state.active_scenarios == []


class TestAuthorizationBoundary:
    def test_the_pipeline_produces_no_trade_decision(self, uptrend_state: MarketState):
        """Risk authorization and the execution gate are not implemented, so
        nothing here may present itself as authorized."""
        result = brain.run(uptrend_state)
        assert not hasattr(result, "trade_decision")
        assert not hasattr(result, "risk_decision")

    def test_actionable_means_survived_analysis_not_authorized(
        self, uptrend_state: MarketState
    ):
        result = brain.run(uptrend_state)
        assert result.is_actionable
        assert result.selected_strategy is not StrategyType.NO_TRADE
        assert result.best_candidate is not None
        # The candidate carries risk *facts*, but no approval.
        assert result.best_candidate.max_loss > 0

    def test_the_result_is_immutable(self, uptrend_state: MarketState):
        result = brain.run(uptrend_state)
        with pytest.raises(ValidationError):
            result.selected_strategy = StrategyType.LONG_CALL  # type: ignore[misc]


class TestSafeDefaults:
    def test_a_contested_market_ends_in_no_trade(self, narrow_rally_state: MarketState):
        result = brain.run(narrow_rally_state)
        assert result.signal.direction is Direction.NEUTRAL
        assert result.selected_strategy is StrategyType.NO_TRADE
        assert result.strike_candidates == []
        assert not result.is_actionable

    def test_an_empty_chain_degrades_to_no_trade_without_raising(
        self, uptrend_state: MarketState
    ):
        empty = uptrend_state.options_state.model_copy(update={"chain": []})
        result = brain.run(uptrend_state.model_copy(update={"options_state": empty}))
        assert result.selected_strategy is StrategyType.NO_TRADE
        assert not result.is_actionable

    def test_a_bare_state_degrades_without_raising(self, uptrend_state: MarketState):
        """Missing history everywhere at once must not throw: spec §29 wants a
        safe stand-down, not an exception."""
        stripped = uptrend_state.model_copy(
            update={
                "index_state": uptrend_state.index_state.model_copy(
                    update={"daily_bars": [], "intraday_bars": [], "opening_range": None}
                ),
                "constituent_state": uptrend_state.constituent_state.model_copy(
                    update={"quotes": [], "weights": {}, "sectors": {}}
                ),
                "options_state": uptrend_state.options_state.model_copy(update={"chain": []}),
            }
        )
        result = brain.run(stripped)
        assert isinstance(result, BrainCycleResult)
        assert result.selected_strategy is StrategyType.NO_TRADE


class TestDeterminism:
    def test_the_same_state_yields_the_same_conclusions(self, uptrend_state: MarketState):
        """Spec §36 requires deterministic reproducibility. Scenario and
        signal ids are fresh per run by design, so the conclusions are
        compared rather than the identifiers."""
        first = brain.run(uptrend_state)
        second = brain.run(uptrend_state)

        assert first.analysis == second.analysis
        assert first.regime == second.regime
        assert first.signal.direction is second.signal.direction
        assert first.signal.score == second.signal.score
        assert first.selected_strategy is second.selected_strategy
        assert [s.kind for s in first.scenarios] == [s.kind for s in second.scenarios]
        assert [s.score for s in first.scenarios] == [s.score for s in second.scenarios]

    def test_independent_brains_agree(self, uptrend_state: MarketState):
        assert QuantitativeBrain().run(uptrend_state).analysis == QuantitativeBrain().run(
            uptrend_state
        ).analysis


class TestMarketCharacterisation:
    @pytest.mark.parametrize(
        "fixture_name,regime,direction",
        [
            ("uptrend_state", MarketRegimeType.TREND_UP, Direction.BULLISH),
            ("downtrend_state", MarketRegimeType.TREND_DOWN, Direction.BEARISH),
            ("range_state", MarketRegimeType.RANGE, Direction.NEUTRAL),
            ("expiry_day_state", MarketRegimeType.EXPIRY, Direction.BULLISH),
        ],
    )
    def test_simulated_markets_are_read_correctly_end_to_end(
        self,
        request,
        fixture_name: str,
        regime: MarketRegimeType,
        direction: Direction,
    ):
        state: MarketState = request.getfixturevalue(fixture_name)
        result = brain.run(state)
        assert result.regime.regime is regime
        assert result.signal.direction is direction

    def test_the_full_reasoning_chain_is_reconstructable(self, uptrend_state: MarketState):
        """Spec §31: every trade must be reconstructable from the record."""
        result = brain.run(uptrend_state)
        assert result.analysis.index.evidence
        assert result.regime.evidence
        assert result.regime.scores
        assert all(s.supporting_evidence for s in result.scenarios)
        assert result.signal.evidence
        assert result.signal.selected_scenario_id in {s.scenario_id for s in result.scenarios}
        assert all(c.rationale for c in result.strategy_candidates)
        assert all(c.rationale for c in result.strike_candidates)


class TestInjection:
    def test_a_stage_can_be_replaced_without_touching_the_others(
        self, uptrend_state: MarketState
    ):
        """What lets a strategy version swap one brain (spec §20)."""
        from index_option_brain.brain.index_brain import DeterministicIndexBrain, IndexBrain
        from index_option_brain.contracts.analysis import IndexAnalysis

        class FlatIndexBrain(IndexBrain):
            def analyze(self, state: MarketState) -> IndexAnalysis:
                return IndexAnalysis(
                    direction=Direction.NEUTRAL,
                    trend_score=0.0,
                    structure_score=0.0,
                    momentum_score=0.0,
                    confidence=0.9,
                )

        custom = QuantitativeBrain(index_brain=FlatIndexBrain())
        result = custom.run(uptrend_state)
        assert result.analysis.index.direction is Direction.NEUTRAL
        assert result.signal.direction is Direction.NEUTRAL

        default = QuantitativeBrain(index_brain=DeterministicIndexBrain())
        assert default.run(uptrend_state).signal.direction is Direction.BULLISH
