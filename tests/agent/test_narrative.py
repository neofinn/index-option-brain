"""The deterministic explanation layer.

Every sentence it writes is assembled from a field that was measured, which
is the property worth protecting: there is no failure mode where the summary
describes a market the engine did not observe. That is also why it does not
need a model, and why the console can render it on every cycle for free.
"""

from __future__ import annotations

from index_option_brain.agent import NarrativeProvider
from index_option_brain.brain.pipeline import QuantitativeBrain
from index_option_brain.contracts.enums import MarketRegimeType, StrategyType
from index_option_brain.contracts.market_state import MarketState

provider = NarrativeProvider()


def describe(state: MarketState):
    result = QuantitativeBrain().run(state)
    return result, provider.describe(
        analysis=result.state.analysis,
        regime=result.regime,
        signal=result.signal,
        strategy=result.selected_strategy,
        candidate=result.best_candidate,
        is_authorized=result.is_authorized,
    )


class TestItDescribesWhatHappened:
    def test_a_trending_market_reads_as_one(self, uptrend_state: MarketState):
        result, assessment = describe(uptrend_state)
        assert result.regime.regime is MarketRegimeType.TREND_UP
        assert "trend up" in assessment.summary.lower()

    def test_the_summary_is_properly_punctuated(self, uptrend_state: MarketState):
        """Read by a person at 09:20, so it has to look like prose."""
        _, assessment = describe(uptrend_state)
        assert assessment.summary.endswith(".")
        sentences = [s.strip() for s in assessment.summary.split(". ") if s.strip()]
        assert len(sentences) > 2
        for sentence in sentences:
            assert sentence[0].isupper(), f"not capitalised: {sentence!r}"

    def test_standing_aside_is_stated_as_a_choice(self, narrow_rally_state: MarketState):
        result, assessment = describe(narrow_rally_state)
        assert result.selected_strategy is StrategyType.NO_TRADE
        assert "stand aside" in assessment.summary

    def test_it_names_the_expected_move(self, uptrend_state: MarketState):
        """The number a buyer reads first."""
        _, assessment = describe(uptrend_state)
        assert "one-sigma move" in assessment.summary

    def test_it_says_whether_premium_is_cheap_or_dear(self, uptrend_state: MarketState):
        _, assessment = describe(uptrend_state)
        assert "premium" in assessment.summary


class TestItNamesWhatItDoesNotKnow:
    """An investigation layer that only reports findings is one that always
    finds something. Naming the gaps is most of the value when the honest
    answer is that the data does not say."""

    def test_a_domain_that_measured_nothing_is_listed_as_unknown(self):
        from tests.events.conftest import state

        # A bare state: no bars, no constituents.
        _, assessment = describe(state())
        joined = " ".join(assessment.unknowns)
        assert "index brain measured nothing" in joined
        assert "constituents brain measured nothing" in joined

    def test_an_unmeasured_domain_contributes_no_evidence(self):
        """Reporting its evidence anyway would present an absence as a
        finding — the failure the regime coverage gate exists to prevent."""
        from tests.events.conftest import state

        _, assessment = describe(state())
        assert not any(p.startswith("index:") for p in assessment.supporting_points)

    def test_uncertainty_is_argued_against_not_asserted(self):
        from tests.events.conftest import state

        _, assessment = describe(state())
        assert assessment.contradicting_points
        assert "UNCERTAIN" in assessment.summary

    def test_no_analysis_at_all_is_reported_as_a_data_problem(self):
        assessment = provider.describe(
            analysis=None, regime=None, signal=None, strategy=StrategyType.NO_TRADE
        )
        assert "data problem" in assessment.summary
        assert assessment.unknowns


class TestProvenance:
    def test_every_claim_cites_a_source(self, uptrend_state: MarketState):
        """So a reader can check it rather than trust it."""
        _, assessment = describe(uptrend_state)
        assert assessment.sources
        assert any("analysis" in source for source in assessment.sources)

    def test_source_confidence_is_carried(self, uptrend_state: MarketState):
        _, assessment = describe(uptrend_state)
        assert any("confidence" in source for source in assessment.sources)

    def test_it_marks_itself_as_the_provider(self, uptrend_state: MarketState):
        _, assessment = describe(uptrend_state)
        assert assessment.provider == "narrative"


class TestItNeedsNoModel:
    async def test_it_works_with_llm_disabled(self, uptrend_state: MarketState):
        """Which is the supported configuration, not a degraded one."""
        from index_option_brain.config.settings import Settings

        assert Settings().llm_enabled is False
        _, assessment = describe(uptrend_state)
        assert assessment.summary

    async def test_it_makes_no_network_call(self, uptrend_state: MarketState):
        """`describe` is synchronous and pure, which is why the console can
        render it on every cycle."""
        import inspect

        assert not inspect.iscoroutinefunction(provider.describe)

    def test_it_carries_no_numeric_field_into_a_decision(self, uptrend_state):
        from index_option_brain.agent.intelligence_provider import AgentAssessment

        _, assessment = describe(uptrend_state)
        assert isinstance(assessment, AgentAssessment)
        for value in assessment.model_dump().values():
            assert not isinstance(value, (int, float)) or isinstance(value, bool)
