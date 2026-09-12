"""Regime Engine behaviour (spec §9).

The property worth protecting is that UNCERTAIN is reachable. An engine that
always names a regime is not classifying, it is guessing — so these tests
cover both the confident cases and the two routes to UNCERTAIN (nothing
clears the floor, and two contradictory readings score alike).
"""

from __future__ import annotations

import pytest

from index_option_brain.brain.config import RegimeEngineConfig
from index_option_brain.brain.pipeline import QuantitativeBrain
from index_option_brain.brain.regime_brain import DeterministicRegimeEngine
from index_option_brain.contracts.analysis import (
    ConstituentAnalysis,
    IndexAnalysis,
    OptionsAnalysis,
    VolatilityAnalysis,
)
from index_option_brain.contracts.enums import (
    BreakoutState,
    Direction,
    IvRegime,
    MarketRegimeType,
)
from index_option_brain.contracts.market_state import MarketState

engine = DeterministicRegimeEngine()


def index_analysis(**overrides) -> IndexAnalysis:
    defaults = {
        "direction": Direction.NEUTRAL,
        "trend_score": 0.0,
        "structure_score": 0.0,
        "momentum_score": 0.0,
        "confidence": 0.8,
    }
    return IndexAnalysis(**{**defaults, **overrides})


def constituent_analysis(**overrides) -> ConstituentAnalysis:
    defaults = {
        "breadth_score": 0.0,
        "participation_score": 0.7,
        "leadership_score": 0.0,
        "concentration_score": 0.2,
        "confidence": 0.8,
    }
    return ConstituentAnalysis(**{**defaults, **overrides})


def options_analysis(**overrides) -> OptionsAnalysis:
    defaults = {
        "call_pressure": 0.3,
        "put_pressure": 0.3,
        "oi_structure_score": 0.0,
        "iv_score": 0.0,
        "liquidity_score": 0.8,
        "confidence": 0.8,
    }
    return OptionsAnalysis(**{**defaults, **overrides})


def volatility_analysis(**overrides) -> VolatilityAnalysis:
    defaults = {
        "regime": IvRegime.NORMAL,
        "expected_move": 400,
        "iv_score": 0.0,
        "expansion_score": 0.0,
        "confidence": 0.8,
        "days_to_expiry": 6.0,
        "iv_percentile": 0.5,
    }
    return VolatilityAnalysis(**{**defaults, **overrides})


def classify(index=None, constituents=None, options=None, volatility=None):
    return engine.classify(
        index or index_analysis(),
        constituents or constituent_analysis(),
        options or options_analysis(),
        volatility or volatility_analysis(),
    )


class TestConfidentClassifications:
    def test_an_aligned_participated_advance_is_trend_up(self):
        state = classify(
            index=index_analysis(trend_score=0.8, structure_score=0.8, momentum_score=0.7),
            constituents=constituent_analysis(breadth_score=0.8, participation_score=0.9),
        )
        assert state.regime is MarketRegimeType.TREND_UP
        assert state.confidence > 0.5

    def test_an_aligned_participated_decline_is_trend_down(self):
        state = classify(
            index=index_analysis(
                trend_score=-0.8, structure_score=-0.8, momentum_score=-0.7
            ),
            constituents=constituent_analysis(breadth_score=-0.8, participation_score=0.9),
        )
        assert state.regime is MarketRegimeType.TREND_DOWN

    def test_a_confirmed_break_is_breakout(self):
        state = classify(
            index=index_analysis(
                trend_score=0.3,
                structure_score=0.3,
                momentum_score=0.3,
                breakout_state=BreakoutState.BREAKOUT,
            ),
            constituents=constituent_analysis(breadth_score=0.6, participation_score=0.9),
            volatility=volatility_analysis(expansion_score=0.7),
        )
        assert state.regime is MarketRegimeType.BREAKOUT

    def test_a_failed_break_is_reversal(self):
        state = classify(
            index=index_analysis(
                trend_score=0.1,
                structure_score=0.0,
                momentum_score=-0.1,
                breakout_state=BreakoutState.FAILED_BREAKOUT,
            )
        )
        assert state.regime is MarketRegimeType.REVERSAL

    def test_a_flat_unbroken_market_is_range(self):
        state = classify(
            index=index_analysis(
                trend_score=0.02,
                structure_score=0.0,
                momentum_score=-0.02,
                volatility_score=0.2,
            )
        )
        assert state.regime is MarketRegimeType.RANGE


class TestExpiryOverride:
    def test_expiry_dominates_the_structural_read(self):
        """On expiry day, pin risk and collapsing time value matter more than
        whatever the trend says — but the structural read is retained as
        evidence rather than discarded."""
        state = classify(
            index=index_analysis(trend_score=0.9, structure_score=0.9, momentum_score=0.9),
            constituents=constituent_analysis(breadth_score=0.9),
            volatility=volatility_analysis(days_to_expiry=0.2),
        )
        assert state.regime is MarketRegimeType.EXPIRY
        assert any("TREND_UP" in item for item in state.evidence)

    def test_expiry_does_not_trigger_days_ahead(self):
        state = classify(
            index=index_analysis(trend_score=0.9, structure_score=0.9, momentum_score=0.9),
            constituents=constituent_analysis(breadth_score=0.9),
            volatility=volatility_analysis(days_to_expiry=5.0),
        )
        assert state.regime is not MarketRegimeType.EXPIRY


class TestUncertainty:
    def test_nothing_clearing_the_floor_reports_uncertain(self):
        """The awkward middle: drifting enough that it isn't a range, not
        enough to be a trend, with no participation to confirm either. Note
        that a genuinely *flat* market is a RANGE, not uncertainty — this
        case is specifically the one where no classification earns its keep.
        """
        state = classify(
            index=index_analysis(
                trend_score=0.3,
                structure_score=0.0,
                momentum_score=0.27,
                volatility_score=0.5,
                confidence=0.2,
            ),
            constituents=constituent_analysis(participation_score=0.2, confidence=0.2),
        )
        assert state.regime is MarketRegimeType.UNCERTAIN
        assert max(state.scores.values()) < 0.3
        assert any("floor" in item for item in state.evidence)

    def test_contradictory_leaders_scoring_alike_report_uncertain(self):
        """A trend and a reversal scoring within a hair of each other is not a
        classification, it is a coin flip."""
        tight = DeterministicRegimeEngine(
            RegimeEngineConfig(separation_threshold=0.9, min_confidence=0.05)
        )
        state = tight.classify(
            index_analysis(trend_score=0.6, structure_score=0.5, momentum_score=-0.55),
            constituent_analysis(breadth_score=0.4),
            options_analysis(),
            volatility_analysis(),
        )
        assert state.regime is MarketRegimeType.UNCERTAIN
        assert any("contradictory" in item for item in state.evidence)

    def test_compatible_leaders_are_not_treated_as_ambiguous(self):
        """TREND_UP and BREAKOUT describe the same market from two angles, so
        one narrowly beating the other is still a classification."""
        state = classify(
            index=index_analysis(
                trend_score=0.7,
                structure_score=0.7,
                momentum_score=0.7,
                breakout_state=BreakoutState.BREAKOUT,
            ),
            constituents=constituent_analysis(breadth_score=0.7, participation_score=0.9),
        )
        assert state.regime in (MarketRegimeType.TREND_UP, MarketRegimeType.BREAKOUT)
        assert state.regime is not MarketRegimeType.UNCERTAIN


class TestVolatilityAxis:
    def test_high_volatility_does_not_displace_a_clean_trend(self):
        """Volatility level is a different axis from structure: a trend can be
        a volatile trend, and reporting HIGH_VOLATILITY there would discard
        the more actionable read."""
        state = classify(
            index=index_analysis(
                trend_score=0.85,
                structure_score=0.85,
                momentum_score=0.8,
                volatility_score=0.95,
            ),
            constituents=constituent_analysis(breadth_score=0.8, participation_score=0.9),
            volatility=volatility_analysis(iv_percentile=0.95),
        )
        assert state.regime is MarketRegimeType.TREND_UP

    def test_high_volatility_surfaces_when_structure_is_silent(self):
        state = classify(
            index=index_analysis(
                trend_score=0.05,
                structure_score=0.0,
                momentum_score=0.0,
                volatility_score=0.95,
                breakout_state=BreakoutState.BREAKOUT,
            ),
            volatility=volatility_analysis(iv_percentile=0.97),
        )
        assert state.regime in (
            MarketRegimeType.HIGH_VOLATILITY,
            MarketRegimeType.BREAKOUT,
        )


class TestOutputShape:
    def test_all_candidate_scores_are_retained_for_review(self):
        state = classify()
        assert set(state.scores) == {r.value for r in MarketRegimeType}

    def test_a_directional_regime_states_what_would_invalidate_it(self):
        state = classify(
            index=index_analysis(
                trend_score=0.8,
                structure_score=0.8,
                momentum_score=0.8,
                support_levels=[24000],
            ),
            constituents=constituent_analysis(breadth_score=0.8, participation_score=0.9),
        )
        assert state.regime is MarketRegimeType.TREND_UP
        assert state.invalidations

    @pytest.mark.parametrize(
        "fixture_name,expected",
        [
            ("uptrend_state", MarketRegimeType.TREND_UP),
            ("downtrend_state", MarketRegimeType.TREND_DOWN),
            ("range_state", MarketRegimeType.RANGE),
            ("expiry_day_state", MarketRegimeType.EXPIRY),
        ],
    )
    def test_end_to_end_regimes_match_the_simulated_market(
        self, request, fixture_name: str, expected: MarketRegimeType
    ):
        state: MarketState = request.getfixturevalue(fixture_name)
        result = QuantitativeBrain().run(state)
        assert result.regime.regime is expected


class TestDataSufficiency:
    """The third route to UNCERTAIN: there was nothing to classify.

    This was found by running the engine against the live NSE feed, which
    serves a chain and India VIX but no historical bars. With no bars the
    index analysis reports every score as 0.0, and the engine happily returned
    RANGE at 1.00 with 0.58 confidence — a confident classification of a
    market nothing had measured. `IndexAnalysis` uses 0.0 for both "measured
    as neutral" and "not measured", and that conflation is what makes an empty
    analysis dangerous rather than merely useless.
    """

    def test_no_index_coverage_yields_uncertain(self):
        state = classify(index=index_analysis(confidence=0.0))
        assert state.regime is MarketRegimeType.UNCERTAIN
        assert state.confidence == 0.0

    def test_the_reason_names_coverage_not_flatness(self):
        """An operator reading "RANGE" acts differently than one reading "no
        data", so the evidence has to distinguish them."""
        state = classify(index=index_analysis(confidence=0.0))
        joined = " ".join(state.evidence)
        assert "coverage" in joined
        assert "unmeasured" in joined

    def test_without_the_gate_zero_coverage_classifies_confidently(self):
        """What the gate actually prevents. With the floor disabled, an
        analysis that measured nothing still produces RANGE at full score —
        because the RANGE candidate reads a composite of 0.0 as perfect
        flatness. Fixing that candidate alone would not be enough either: the
        same unmeasured zeros feed LOW_VOLATILITY, which reads a volatility
        score of 0.0 as perfectly calm. Hence a gate on coverage rather than a
        patch on one candidate."""
        ungated = DeterministicRegimeEngine(
            RegimeEngineConfig(min_index_confidence=0.0)
        )
        state = ungated.classify(
            index_analysis(confidence=0.0, volatility_score=0.0),
            constituent_analysis(),
            options_analysis(),
            volatility_analysis(iv_percentile=None),
        )
        assert state.regime is MarketRegimeType.RANGE
        assert state.scores["RANGE"] >= 0.9
        assert state.confidence > 0.0

    def test_thin_coverage_also_yields_uncertain(self):
        """A handful of bars is not a measured market. Index confidence folds
        in bar sufficiency, so a thin series lands below the floor too."""
        state = classify(index=index_analysis(confidence=0.05))
        assert state.regime is MarketRegimeType.UNCERTAIN

    def test_a_genuinely_flat_measured_market_is_still_range(self):
        """The gate must not swallow the legitimate case: a market with full
        bar coverage and nothing happening is a range, and saying so is the
        whole point of having a RANGE label."""
        state = classify(
            index=index_analysis(confidence=0.4, breakout_state=BreakoutState.NONE)
        )
        assert state.regime is MarketRegimeType.RANGE

    def test_expiry_survives_missing_index_data(self):
        """Expiry is a calendar fact read off the option chain, not a
        measurement of structure, so it stays valid when structure does not.
        The live feed serves the chain even when it serves no bars."""
        state = classify(
            index=index_analysis(confidence=0.0),
            volatility=volatility_analysis(days_to_expiry=0.5),
        )
        assert state.regime is MarketRegimeType.EXPIRY

    def test_the_floor_is_configurable(self):
        lenient = DeterministicRegimeEngine(
            RegimeEngineConfig(min_index_confidence=0.0)
        )
        state = lenient.classify(
            index_analysis(confidence=0.0),
            constituent_analysis(),
            options_analysis(),
            volatility_analysis(),
        )
        assert state.regime is MarketRegimeType.RANGE

    def test_scores_are_still_reported_for_inspection(self):
        """Refusing to classify is not refusing to explain. The candidate
        scores stay attached so an operator can see what the engine would have
        said and why it declined."""
        state = classify(index=index_analysis(confidence=0.0))
        assert state.scores
        assert set(state.scores) >= {"RANGE", "TREND_UP", "TREND_DOWN"}
