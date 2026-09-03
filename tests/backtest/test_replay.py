"""Replay engine behaviour.

The lookahead tests are the load-bearing ones. Every other number this
module produces is worthless if the brain can see the future, and lookahead
does not announce itself — it shows up as an unusually good result.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from index_option_brain.backtest.replay import (
    DailyReplayEngine,
    DirectionStats,
    evaluate,
    state_from_bars,
)
from index_option_brain.contracts.enums import Direction
from index_option_brain.contracts.instruments import Bar

START = datetime(2026, 1, 1, tzinfo=UTC)


def series(closes: list[float]) -> list[Bar]:
    """A daily series with a small range around each close."""
    return [
        Bar(
            timestamp=START + timedelta(days=i),
            open=Decimal(str(c)),
            high=Decimal(str(c + 30)),
            low=Decimal(str(c - 30)),
            close=Decimal(str(c)),
            volume=100_000,
        )
        for i, c in enumerate(closes)
    ]


class TestNoLookahead:
    def test_the_current_bar_is_the_quote_and_not_a_prior_bar(self) -> None:
        """IndexState.daily_bars holds *completed prior* sessions. Appending
        the session being decided on would let every previous-day level
        silently include today."""
        bars = series([100.0, 101.0, 102.0, 103.0])
        state = state_from_bars(index_symbol="NIFTY", bars=bars)

        assert state.index_state.quote.ltp == Decimal("103.0")
        assert state.index_state.quote.previous_close == Decimal("102.0")
        assert len(state.index_state.daily_bars) == 3
        assert state.index_state.daily_bars[-1].close == Decimal("102.0")

    def test_a_future_step_change_does_not_move_the_decision_before_it(self) -> None:
        """The decisive test.

        Two series identical up to bar 60 and wildly different after. Every
        decision taken at or before bar 60 must be identical between them —
        if any differs, the brain saw the future.
        """
        flat = [24000.0 + (i % 5) * 10 for i in range(90)]
        crash = list(flat)
        for i in range(61, 90):
            crash[i] = 24000.0 - (i - 60) * 200  # a violent decline after bar 60

        engine = DailyReplayEngine(warmup=40, horizons=(1,))
        a = engine.run("NIFTY", series(flat))
        b = engine.run("NIFTY", series(crash))

        shared = [c for c in a if c.bars_seen <= 61]
        other = {c.bars_seen: c for c in b}
        assert shared, "the test needs at least one comparable cycle"
        for cycle in shared:
            twin = other[cycle.bars_seen]
            assert cycle.view is twin.view
            assert cycle.view_score == pytest.approx(twin.view_score)
            assert cycle.regime.regime is twin.regime.regime
            assert cycle.signal.score == pytest.approx(twin.signal.score)

    def test_forward_returns_come_from_bars_the_brain_never_saw(self) -> None:
        bars = series([100.0 + i for i in range(40)])
        engine = DailyReplayEngine(warmup=30, horizons=(1, 3))
        cycles = engine.run("NIFTY", bars)

        first = cycles[0]
        # bars_seen bars were visible, so index = bars_seen - 1.
        index = first.bars_seen - 1
        expected = (
            float(bars[index + 1].close) - float(bars[index].close)
        ) / float(bars[index].close) * 100.0
        assert first.forward_returns[1] == pytest.approx(expected)

    def test_a_horizon_past_the_end_is_absent_not_zero(self) -> None:
        """A truncated tail must not read as a run of flat outcomes."""
        bars = series([100.0 + i for i in range(35)])
        cycles = DailyReplayEngine(warmup=30, horizons=(1, 10)).run("NIFTY", bars)

        assert cycles, "warmup must leave some cycles"
        assert 10 not in cycles[-1].forward_returns
        assert 1 not in cycles[-1].forward_returns


class TestVixAlignment:
    def test_a_misaligned_vix_series_is_refused(self) -> None:
        """An off-by-one here reads yesterday's volatility into today's
        decision, and would look like signal."""
        bars = series([100.0 + i for i in range(40)])
        with pytest.raises(ValueError, match="cannot align"):
            DailyReplayEngine(warmup=30).run("NIFTY", bars, vix_bars=bars[:-1])

    def test_vix_reaches_the_volatility_state_as_vix_not_as_atm_iv(self) -> None:
        """VIX is a 30-day interpolated index, not the ATM implied volatility
        of the weekly this system trades. Passing it off as `atm_iv` would let
        every richness comparison downstream read a different measurement
        under a familiar name."""
        bars = series([100.0, 101.0, 102.0])
        vix = series([12.0, 13.0, 14.0])
        state = state_from_bars(index_symbol="NIFTY", bars=bars, vix_bars=vix)

        assert state.volatility_state.india_vix == 14.0
        assert state.volatility_state.india_vix_previous_close == 13.0
        assert state.volatility_state.atm_iv is None

    def test_no_vix_leaves_volatility_unmeasured(self) -> None:
        state = state_from_bars(index_symbol="NIFTY", bars=series([100.0, 101.0]))
        assert state.volatility_state.india_vix is None


class TestReporting:
    def test_a_hit_rate_carries_its_standard_error(self) -> None:
        """Without one, a 64.7% hit rate at n=17 reads as decisive."""
        stats = DirectionStats(
            label="bullish", count=17, mean_return=0.457,
            median_return=0.573, hit_rate=0.647,
        )
        se = stats.hit_rate_standard_error
        assert se is not None
        assert se == pytest.approx(0.1159, abs=1e-3)
        # 14 points over a 50.7% base rate is about one se, so not evidence.
        assert stats.beats(0.507, sigmas=2.0) is False

    def test_a_neutral_group_has_no_hit_rate(self) -> None:
        """"Correct" has no meaning for a decision that took no view."""
        stats = DirectionStats(
            label="neutral", count=200, mean_return=-0.02,
            median_return=-0.02, hit_rate=None,
        )
        assert stats.hit_rate_standard_error is None
        assert stats.beats(0.5) is None

    def test_the_base_rate_comes_from_the_whole_series_when_given(self) -> None:
        """A base rate drawn from the replayed window is already conditioned
        on the warm-up having passed."""
        bars = series([100.0 + i for i in range(60)])  # monotonically rising
        cycles = DailyReplayEngine(warmup=30, horizons=(1,)).run("NIFTY", bars)
        report = evaluate(cycles, horizon=1, all_bars=bars)

        assert report.base_rate_up == pytest.approx(1.0)

    def test_the_smallest_directional_sample_is_reported(self) -> None:
        bars = series([24000.0 + (i % 7) * 15 for i in range(80)])
        cycles = DailyReplayEngine(warmup=40, horizons=(1,)).run("NIFTY", bars)
        report = evaluate(cycles, horizon=1, all_bars=bars)

        assert report.smallest_directional_sample >= 0
        assert report.sessions == len(
            [c for c in cycles if 1 in c.forward_returns]
        )

    def test_an_empty_replay_reports_nothing_rather_than_zeroes(self) -> None:
        report = evaluate([], horizon=1)
        assert report.sessions == 0
        assert report.base_rate_up is None
        assert report.mean_session_return is None
        assert report.edge_over_base_rate() is None
        assert report.no_view_share is None


class TestTradeableVersusAnalysis:
    def test_the_tradeable_signal_and_the_analysis_view_are_separate(self) -> None:
        """In a chainless replay the Scenario Engine's NO_TRADE case wins on
        the evidence that there is no chain — correct, and total. The two
        readings must not be conflated, or the harness measures the missing
        data instead of the strategy."""
        bars = series([24000.0 - i * 40 for i in range(70)])  # a clear decline
        cycles = DailyReplayEngine(warmup=40, horizons=(1,)).run("NIFTY", bars)

        assert cycles
        # No chain, so nothing is tradeable.
        assert all(c.tradeable_direction is Direction.NEUTRAL for c in cycles)
        # The report says so explicitly rather than leaving it inferred.
        assert evaluate(cycles, horizon=1, all_bars=bars).tradeable_views == 0

    def test_a_view_below_the_score_floor_is_not_counted(self) -> None:
        """Every scenario carries some score; without a floor this evaluates
        noise."""
        bars = series([24000.0 + (i % 3) * 5 for i in range(70)])
        strict = DailyReplayEngine(warmup=40, horizons=(1,), min_view_score=0.99)
        assert all(not c.took_a_view for c in strict.run("NIFTY", bars))


class TestStateConstruction:
    def test_one_bar_cannot_describe_a_session_change(self) -> None:
        with pytest.raises(ValueError, match="at least two bars"):
            state_from_bars(index_symbol="NIFTY", bars=series([100.0]))

    def test_the_session_is_marked_closed(self) -> None:
        """A daily bar describes a whole session; by the time it exists the
        session is over. Marking it ACTIVE would put the Scenario Engine in
        an intraday frame it has no intraday data for."""
        state = state_from_bars(index_symbol="NIFTY", bars=series([100.0, 101.0]))
        assert str(state.session_state) == "CLOSED"
