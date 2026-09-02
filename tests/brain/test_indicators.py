"""Indicator math, checked against hand-computable values.

These are the numeric foundation every brain sits on, so they are tested
against arithmetic rather than against "looks about right" — a subtly wrong
ATR would shift breakout detection, position sizing, and expected moves at
once, and would be very hard to spot downstream.
"""

from __future__ import annotations

import math

import pytest

from index_option_brain.brain import indicators as ind


class TestBasics:
    def test_mean_and_stdev(self):
        assert ind.mean([1.0, 2.0, 3.0]) == 2.0
        assert ind.stdev([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]) == pytest.approx(
            2.13809, rel=1e-4
        )

    def test_insufficient_data_returns_none_rather_than_a_default(self):
        """A fabricated indicator reading is indistinguishable from a real one
        downstream, so absence must be explicit."""
        assert ind.mean([]) is None
        assert ind.stdev([1.0]) is None
        assert ind.sma([1.0, 2.0], 5) is None
        assert ind.ema([1.0, 2.0], 5) is None
        assert ind.rsi([1.0] * 5, 14) is None
        assert ind.atr([1.0], [1.0], [1.0], 14) is None
        assert ind.rate_of_change([1.0], 5) is None
        assert ind.linreg_slope([1.0]) is None
        assert ind.percentile_rank([], 1.0) is None

    def test_sma_uses_only_the_trailing_window(self):
        assert ind.sma([100.0, 1.0, 2.0, 3.0], 3) == 2.0

    def test_ema_weights_recent_values_more_heavily(self):
        rising = list(range(1, 21))
        values = [float(v) for v in rising]
        assert ind.ema(values, 5) > ind.sma(values, 5)

    def test_clamp_and_squash_bounds(self):
        assert ind.clamp(5.0) == 1.0
        assert ind.clamp(-5.0) == -1.0
        assert ind.clamp(0.3, 0.0, 1.0) == 0.3
        assert ind.squash(0.0, 1.0) == 0.0
        assert ind.squash(1.0, 1.0) == pytest.approx(math.tanh(1.0))
        assert -1.0 < ind.squash(1e6, 1.0) <= 1.0
        assert ind.squash(1.0, 0.0) == 0.0  # non-positive scale is inert


class TestRsi:
    def test_all_gains_saturates_high(self):
        assert ind.rsi([float(v) for v in range(1, 30)], 14) == 100.0

    def test_all_losses_saturates_low(self):
        assert ind.rsi([float(v) for v in range(30, 1, -1)], 14) == 0.0

    def test_flat_series_is_neutral(self):
        assert ind.rsi([100.0] * 20, 14) == 50.0

    def test_known_value(self):
        # Alternating +2/-1 over 14 changes: avg gain 1.0, avg loss 0.5, RS 2.
        values = [100.0]
        for i in range(14):
            values.append(values[-1] + (2.0 if i % 2 == 0 else -1.0))
        assert ind.rsi(values, 14) == pytest.approx(100 - 100 / 3, rel=1e-6)


class TestAtr:
    def test_true_range_uses_the_previous_close(self):
        # A gap up: the true range spans from the prior close, not the bar low.
        highs = [100.0, 120.0]
        lows = [99.0, 118.0]
        closes = [99.5, 119.0]
        assert ind.true_ranges(highs, lows, closes) == [pytest.approx(20.5)]

    def test_atr_is_the_mean_of_the_trailing_true_ranges(self):
        highs = [10.0, 11.0, 12.0, 13.0]
        lows = [9.0, 10.0, 11.0, 12.0]
        closes = [9.5, 10.5, 11.5, 12.5]
        # Each true range is 1.5 (high-low = 1, |high - prev close| = 1.5).
        assert ind.atr(highs, lows, closes, 3) == pytest.approx(1.5)


class TestSlopeAndRoc:
    def test_slope_of_a_straight_line_is_its_gradient(self):
        assert ind.linreg_slope([0.0, 2.0, 4.0, 6.0]) == pytest.approx(2.0)

    def test_slope_of_a_flat_line_is_zero(self):
        assert ind.linreg_slope([5.0] * 10) == pytest.approx(0.0)

    def test_rate_of_change_is_fractional(self):
        assert ind.rate_of_change([100.0, 0.0, 0.0, 110.0], 3) == pytest.approx(0.10)


class TestVwap:
    def test_volume_weighting(self):
        assert ind.vwap([10.0, 20.0], [1.0, 3.0]) == pytest.approx(17.5)

    def test_zero_volume_falls_back_to_the_simple_mean(self):
        assert ind.vwap([10.0, 20.0], [0.0, 0.0]) == pytest.approx(15.0)


class TestPercentileRank:
    def test_rank_within_history(self):
        history = [1.0, 2.0, 3.0, 4.0]
        assert ind.percentile_rank(history, 2.0) == 0.5
        assert ind.percentile_rank(history, 0.0) == 0.0
        assert ind.percentile_rank(history, 10.0) == 1.0


class TestSwingsAndStructure:
    def test_swing_highs_require_strictly_higher_pivots(self):
        highs = [1.0, 2.0, 5.0, 2.0, 1.0]
        assert ind.swing_high_indices(highs, 2) == [2]
        # A plateau is not a pivot: neither shoulder is strictly lower.
        assert ind.swing_high_indices([1.0, 5.0, 5.0, 5.0, 1.0], 2) == []

    def test_swing_lows(self):
        lows = [5.0, 4.0, 1.0, 4.0, 5.0]
        assert ind.swing_low_indices(lows, 2) == [2]

    def test_higher_highs_and_higher_lows_score_positive(self):
        highs = [1, 3, 1, 1, 5, 1, 1, 7, 1, 1]
        lows = [10, 8, 10, 10, 9, 10, 10, 10, 10, 10]
        score = ind.market_structure_score([float(h) for h in highs], [float(x) for x in lows], 2)
        assert score is not None
        assert score > 0

    def test_lower_highs_and_lower_lows_score_negative(self):
        highs = [1, 7, 1, 1, 5, 1, 1, 3, 1, 1]
        lows = [10, 5, 10, 10, 3, 10, 10, 1, 10, 10]
        score = ind.market_structure_score([float(h) for h in highs], [float(x) for x in lows], 2)
        assert score is not None
        assert score < 0

    def test_structure_is_none_without_two_comparable_swings(self):
        assert ind.market_structure_score([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 2) is None


class TestConcentrationAndAlignment:
    def test_even_weights_are_unconcentrated(self):
        assert ind.normalized_hhi([1.0, 1.0, 1.0, 1.0]) == pytest.approx(0.0)

    def test_a_single_dominant_member_is_fully_concentrated(self):
        assert ind.normalized_hhi([10.0, 0.0, 0.0, 0.0]) == pytest.approx(1.0)

    def test_sign_is_ignored_only_magnitude_concentrates(self):
        assert ind.normalized_hhi([-5.0, 5.0]) == pytest.approx(0.0)

    def test_hhi_of_nothing_is_none(self):
        assert ind.normalized_hhi([]) is None
        assert ind.normalized_hhi([0.0, 0.0]) is None

    def test_unanimous_scores_are_fully_aligned(self):
        assert ind.alignment([0.5, 0.7, 0.2]) == pytest.approx(1.0)

    def test_offsetting_scores_are_unaligned(self):
        assert ind.alignment([0.5, -0.5]) == pytest.approx(0.0)

    def test_one_strong_reading_dragging_contradictions_scores_low(self):
        """The distinction that stops a single indicator carrying a trade."""
        unanimous = ind.alignment([0.4, 0.4, 0.4])
        dragged = ind.alignment([0.9, -0.3, -0.3])
        assert dragged < unanimous

    def test_alignment_ignores_missing_components(self):
        assert ind.alignment([0.5, None, 0.5]) == pytest.approx(1.0)
        assert ind.alignment([None, None]) == 0.0


class TestBlend:
    def test_weighted_mean(self):
        assert ind.blend((1.0, 3.0), (0.0, 1.0)) == pytest.approx(0.75)

    def test_missing_components_are_dropped_and_weights_renormalized(self):
        """A partially observable state must degrade smoothly, not be scored
        as though the missing inputs read zero."""
        assert ind.blend((1.0, 1.0), (None, 9.0)) == pytest.approx(1.0)

    def test_all_missing_is_none(self):
        assert ind.blend((None, 1.0), (None, 1.0)) is None
        assert ind.blend() is None
