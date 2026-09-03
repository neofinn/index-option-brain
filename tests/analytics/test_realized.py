"""Realized volatility estimators and the volatility risk premium.

The estimator choice matters in a specific direction, which is what most of
these pin: an estimator that understates realized volatility overstates the
premium, which makes options look expensive — and this system buys them.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from index_option_brain.analytics.realized import (
    VolatilityEstimator,
    close_to_close,
    estimate,
    garman_klass,
    parkinson,
    volatility_risk_premium,
    window_for_tenor,
    yang_zhang,
)
from index_option_brain.contracts.instruments import Bar

START = datetime(2026, 1, 1, tzinfo=UTC)


def bars(
    closes: list[float], *, range_pct: float = 0.5, gap_pct: float = 0.0
) -> list[Bar]:
    """A series with a controllable intraday range and overnight gap.

    `gap_pct` moves each open away from the previous close, which is the
    variance the range estimators cannot see.
    """
    out: list[Bar] = []
    previous: float | None = None
    for i, close in enumerate(closes):
        open_ = close if previous is None else previous * (1 + gap_pct / 100.0)
        high = max(open_, close) * (1 + range_pct / 200.0)
        low = min(open_, close) * (1 - range_pct / 200.0)
        out.append(
            Bar(
                timestamp=START + timedelta(days=i),
                open=Decimal(str(round(open_, 4))),
                high=Decimal(str(round(high, 4))),
                low=Decimal(str(round(low, 4))),
                close=Decimal(str(round(close, 4))),
                volume=1000,
            )
        )
        previous = close
    return out


def wobble(n: int, *, amplitude: float = 1.0, base: float = 24000.0) -> list[float]:
    return [base * (1 + amplitude / 100.0 * math.sin(i * 1.7)) for i in range(n)]


class TestSampleRequirements:
    def test_too_few_bars_yields_none_not_zero(self) -> None:
        """A realized volatility of zero is a market that did not move; an
        unmeasured one is a window nobody looked at."""
        assert close_to_close(bars(wobble(3))) is None
        assert parkinson(bars(wobble(2))) is None
        assert estimate(bars(wobble(2)), window=10) is None

    def test_an_empty_series_yields_none(self) -> None:
        assert estimate([], window=20) is None

    def test_a_flat_market_is_zero_volatility_and_reported_as_absent(self) -> None:
        """Zero is not a usable realized volatility for a ratio, so estimate
        declines rather than returning something that divides badly."""
        flat = bars([24000.0] * 30, range_pct=0.0)
        assert estimate(flat, window=20) is None


class TestMalformedBars:
    def test_a_crossed_bar_is_dropped_rather_than_producing_nan(self) -> None:
        """A high below its low is a feed artifact, and a log of it
        propagates a NaN through every downstream comparison."""
        series = bars(wobble(30))
        broken = series[10].model_copy(update={"high": Decimal(1), "low": Decimal(2)})
        series[10] = broken
        result = estimate(series, window=30)
        assert result is not None
        assert math.isfinite(result.value)
        assert result.bars_used == 29

    def test_a_zero_price_is_dropped(self) -> None:
        series = bars(wobble(30))
        series[5] = series[5].model_copy(update={"low": Decimal(0)})
        result = estimate(series, window=30)
        assert result is not None
        assert result.bars_used == 29


def gap_only(n: int, *, gap_pct: float = 0.9) -> list[Bar]:
    """A market that moves *only* between sessions.

    Each session opens where it closes, so the intraday range is zero and a
    range estimator must report no volatility at all — while the index has
    in fact moved `gap_pct` every night. This isolates the variance
    Parkinson and Garman-Klass are structurally unable to see.
    """
    out: list[Bar] = []
    level = 24000.0
    for i in range(n):
        level *= 1 + (gap_pct if i % 2 else -gap_pct) / 100.0
        price = Decimal(str(round(level, 4)))
        out.append(
            Bar(
                timestamp=START + timedelta(days=i),
                open=price,
                high=price,
                low=price,
                close=price,
                volume=1000,
            )
        )
    return out


class TestGapBlindness:
    """The measurement that decided the default estimator.

    On 41 real NIFTY sessions to 2 Sep 2026, Parkinson read 6.76% where
    close-to-close read 11.23% — a 40% understatement, because an Indian
    index does much of its moving between sessions. These tests isolate that
    mechanism synthetically.
    """

    def test_a_range_estimator_reports_no_volatility_in_a_gap_only_market(
        self,
    ) -> None:
        """The market moves 0.9% every night and not at all during the day.
        Parkinson measures the day, so it sees a dead market.

        0.9% nightly annualizes to about 14% (0.9 x sqrt(252)), which is the
        figure the gap-aware estimators must recover.
        """
        series = gap_only(40)

        assert parkinson(series) == pytest.approx(0.0, abs=1e-9)
        assert garman_klass(series) == pytest.approx(0.0, abs=1e-9)

        # The move is real, and the gap-aware estimators find it.
        assert close_to_close(series) == pytest.approx(14.3, rel=0.05)
        assert yang_zhang(series) == pytest.approx(14.3, rel=0.10)

    def test_the_bias_runs_in_the_direction_that_matters(self) -> None:
        """Understating realized overstates the premium, which makes options
        look expensive — and this system buys them."""
        series = gap_only(40)
        # Implied sits below the ~14% the index is actually delivering, so
        # an honest measurement calls this premium cheap.
        cheap_looking = volatility_risk_premium(
            implied=10.0,
            bars=series,
            days_to_expiry=30,
            estimator=VolatilityEstimator.CLOSE_TO_CLOSE,
        )
        rich_looking = volatility_risk_premium(
            implied=10.0,
            bars=series,
            days_to_expiry=30,
            estimator=VolatilityEstimator.PARKINSON,
        )
        assert cheap_looking is not None
        # Parkinson cannot even produce a measurement here, which is the
        # honest outcome: zero volatility is not a usable denominator.
        assert rich_looking is None
        assert cheap_looking.premium < 0  # correctly reads as cheap

    def test_the_default_estimator_accounts_for_gaps(self) -> None:
        """Parkinson would understate, which overstates the premium and
        makes options look expensive — the wrong direction for a buyer."""
        series = bars(wobble(40), gap_pct=0.3)
        chosen = estimate(series, window=30)
        assert chosen is not None
        assert chosen.estimator is VolatilityEstimator.YANG_ZHANG

    def test_a_short_window_falls_back_to_close_to_close_not_parkinson(self) -> None:
        """Close-to-close spans the gap by construction. Parkinson cannot."""
        chosen = estimate(bars(wobble(6)), window=6)
        assert chosen is not None
        assert chosen.estimator is VolatilityEstimator.CLOSE_TO_CLOSE

    def test_garman_klass_is_available_but_never_the_default(self) -> None:
        """It answers a real question — how far price moved during the
        session — which is not the question the premium asks."""
        series = bars(wobble(40))
        assert garman_klass(series) is not None
        assert estimate(series, window=30).estimator is not (
            VolatilityEstimator.GARMAN_KLASS
        )


class TestHorizonMatching:
    def test_the_window_follows_the_option_tenor(self) -> None:
        """A five-day option's implied volatility against ninety sessions of
        realized is what this replaces."""
        assert window_for_tenor(5.2) == 10  # floored
        assert window_for_tenor(30) == 21
        assert window_for_tenor(120) == 60  # capped

    def test_an_unknown_tenor_uses_the_floor_not_the_whole_series(self) -> None:
        assert window_for_tenor(None) == 10
        assert window_for_tenor(0) == 10

    def test_only_the_window_is_measured(self) -> None:
        calm = wobble(60, amplitude=0.2)
        violent = wobble(20, amplitude=3.0)
        series = bars(calm + violent)

        recent = estimate(series, window=20)
        whole = estimate(series, window=80)
        assert recent is not None and whole is not None
        # The recent window sits in the violent stretch, so it must read
        # higher than the blend of both.
        assert recent.value > whole.value

    def test_the_measurement_says_how_it_was_made(self) -> None:
        """"Realized volatility is 11%" is not a fact on its own."""
        result = estimate(bars(wobble(40)), window=20)
        assert result is not None
        assert result.estimator
        assert result.window == 20
        assert result.is_horizon_matched


class TestVolatilityRiskPremium:
    def test_a_calm_tape_under_a_high_surface_is_a_rich_premium(self) -> None:
        calm = bars(wobble(40, amplitude=0.15))
        premium = volatility_risk_premium(
            implied=18.0, bars=calm, days_to_expiry=7
        )
        assert premium is not None
        assert premium.premium > 0
        assert premium.favours_buying is False
        assert premium.score() > 0

    def test_a_violent_tape_under_a_low_surface_favours_buying(self) -> None:
        """The case a seller-framed rule of thumb skips past, and the one
        this system's mandate cares about."""
        violent = bars(wobble(40, amplitude=3.0))
        premium = volatility_risk_premium(
            implied=9.0, bars=violent, days_to_expiry=7
        )
        assert premium is not None
        assert premium.premium < 0
        assert premium.favours_buying is True
        assert premium.score() < 0

    def test_the_score_is_bounded(self) -> None:
        violent = bars(wobble(40, amplitude=8.0))
        premium = volatility_risk_premium(
            implied=5.0, bars=violent, days_to_expiry=7
        )
        assert premium is not None
        assert premium.score() == -1.0

    def test_an_unmeasurable_side_yields_none_rather_than_a_default(self) -> None:
        """A premium computed against an assumed realized volatility is a
        statement about the assumption."""
        series = bars(wobble(40))
        assert volatility_risk_premium(
            implied=None, bars=series, days_to_expiry=7
        ) is None
        assert volatility_risk_premium(
            implied=12.0, bars=[], days_to_expiry=7
        ) is None
        assert volatility_risk_premium(
            implied=0.0, bars=series, days_to_expiry=7
        ) is None

    def test_the_description_names_both_horizons(self) -> None:
        premium = volatility_risk_premium(
            implied=18.0, bars=bars(wobble(40, amplitude=0.15)), days_to_expiry=7
        )
        assert premium is not None
        text = premium.describe()
        assert "session realized" in text
        assert "rich" in text


class TestEstimatorAgreement:
    def test_the_gap_free_case_is_where_they_agree(self) -> None:
        """Sanity: without gaps the estimators should be in the same
        neighbourhood, which is what makes the gapped divergence meaningful
        rather than a bug in one of them."""
        series = bars(wobble(60, amplitude=1.0), range_pct=0.8, gap_pct=0.0)
        values = [
            close_to_close(series),
            parkinson(series),
            garman_klass(series),
            yang_zhang(series),
        ]
        assert all(v is not None for v in values)
        assert max(values) / min(values) < 4.0  # type: ignore[type-var]
