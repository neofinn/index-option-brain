"""Constituent Brain behaviour (spec §6).

The behaviour that matters most here is the one the spec calls out
explicitly: distinguishing broad participation from an index move driven by
a handful of heavyweights.
"""

from __future__ import annotations

from index_option_brain.brain.constituent_brain import DeterministicConstituentBrain
from index_option_brain.contracts.market_state import MarketState

brain = DeterministicConstituentBrain()


class TestBreadth:
    def test_a_broad_rally_reads_positive_breadth(self, uptrend_state: MarketState):
        analysis = brain.analyze(uptrend_state)
        assert analysis.breadth_score > 0.3
        assert analysis.advances > analysis.declines
        assert analysis.weighted_change_pct > 0

    def test_a_broad_decline_reads_negative_breadth(self, downtrend_state: MarketState):
        analysis = brain.analyze(downtrend_state)
        assert analysis.breadth_score < -0.3
        assert analysis.declines > analysis.advances

    def test_advance_decline_counts_cover_every_observed_name(
        self, uptrend_state: MarketState
    ):
        analysis = brain.analyze(uptrend_state)
        total = analysis.advances + analysis.declines + analysis.unchanged
        assert total == len(uptrend_state.constituent_state.quotes)


class TestParticipationVersusConcentration:
    def test_a_broad_rally_has_high_participation(self, uptrend_state: MarketState):
        analysis = brain.analyze(uptrend_state)
        assert analysis.participation_score > 0.6

    def test_a_heavyweight_driven_rally_has_low_participation(
        self, narrow_rally_state: MarketState
    ):
        """The core discrimination: the index is up, but most constituents are
        not, so the move is not confirmed by participation."""
        analysis = brain.analyze(narrow_rally_state)
        assert narrow_rally_state.index_state.quote.change_pct > 0
        assert analysis.breadth_score < 0
        assert analysis.participation_score < 0.5

    def test_participation_is_measured_against_the_index_not_its_own_aggregate(
        self, narrow_rally_state: MarketState
    ):
        """Measuring against the contribution sum would be circular — the
        observed names would always agree with their own total."""
        analysis = brain.analyze(narrow_rally_state)
        broad_agreement = (
            max(analysis.advances, analysis.declines)
            / len(narrow_rally_state.constituent_state.quotes)
        )
        assert analysis.participation_score < broad_agreement

    def test_concentration_is_bounded_and_reported(self, uptrend_state: MarketState):
        analysis = brain.analyze(uptrend_state)
        assert 0.0 <= analysis.concentration_score <= 1.0


class TestContributionsAndSectors:
    def test_contributions_are_weight_adjusted(self, uptrend_state: MarketState):
        analysis = brain.analyze(uptrend_state)
        weights = uptrend_state.constituent_state.weights
        for symbol, contribution in analysis.contributions.items():
            change = next(
                float(q.change_pct)
                for q in uptrend_state.constituent_state.quotes
                if q.symbol == symbol
            )
            assert contribution == weights[symbol] * change / 100.0

    def test_contributors_and_detractors_are_signed_correctly(
        self, narrow_rally_state: MarketState
    ):
        analysis = brain.analyze(narrow_rally_state)
        for symbol in analysis.top_contributors:
            assert analysis.contributions[symbol] > 0
        for symbol in analysis.top_detractors:
            assert analysis.contributions[symbol] < 0

    def test_sector_scores_cover_the_observed_sectors(self, uptrend_state: MarketState):
        analysis = brain.analyze(uptrend_state)
        expected = set(uptrend_state.constituent_state.sectors.values())
        assert set(analysis.sector_scores) <= expected
        assert analysis.sector_scores

    def test_weight_coverage_is_reported(self, uptrend_state: MarketState):
        analysis = brain.analyze(uptrend_state)
        assert 0.0 < analysis.weight_coverage <= 1.0


class TestDegradedInput:
    def test_no_quotes_yields_zero_confidence_not_an_error(
        self, uptrend_state: MarketState
    ):
        empty = uptrend_state.constituent_state.model_copy(update={"quotes": []})
        analysis = brain.analyze(uptrend_state.model_copy(update={"constituent_state": empty}))
        assert analysis.confidence == 0.0
        assert analysis.breadth_score == 0.0
        assert analysis.evidence

    def test_missing_weights_still_produce_breadth(self, uptrend_state: MarketState):
        """Breadth only needs prices; contribution needs weights. Losing the
        weights must not take the whole analysis down with it."""
        unweighted = uptrend_state.constituent_state.model_copy(update={"weights": {}})
        analysis = brain.analyze(
            uptrend_state.model_copy(update={"constituent_state": unweighted})
        )
        assert analysis.breadth_score > 0
        assert analysis.contributions == {}
        assert analysis.weight_coverage == 0.0
