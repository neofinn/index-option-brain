"""Projecting what an option does when the index moves.

The greeks are rates of change; a trader needs the change. These tests pin
the two things that make the projection trustworthy: the exact repricing is
exact, and the greek decomposition is kept honest about where it breaks.
"""

from __future__ import annotations

import pytest

from index_option_brain.analytics.pricing import price_option
from index_option_brain.analytics.reaction import (
    Scenario,
    gap_scenarios,
    project,
    volatility_crush,
)
from index_option_brain.contracts.enums import OptionType

SPOT = 23_914.0
YEARS = 5 / 365
IV = 6.2


def call(strike: float, scenario: Scenario, *, iv: float = IV):
    return project(
        spot=SPOT, strike=strike, years=YEARS, iv_percent=iv,
        option_type=OptionType.CE, scenario=scenario,
    )


class TestTheRepricingIsExact:
    def test_a_flat_scenario_only_charges_time(self):
        reaction = call(23_900, Scenario("flat", days_elapsed=1.0))
        assert reaction.change < 0
        assert reaction.delta_contribution == 0.0
        assert reaction.gamma_contribution == 0.0

    def test_the_new_price_matches_a_direct_repricing(self):
        """The exact leg is not an approximation of anything."""
        scenario = Scenario("up", spot_change=119.0, days_elapsed=1.0)
        reaction = call(23_900, scenario)
        direct = price_option(
            spot=SPOT + 119.0,
            strike=23_900,
            years=YEARS - 1 / 365,
            iv=IV / 100,
            option_type=OptionType.CE,
        ).price
        assert reaction.price_after == pytest.approx(direct, abs=1e-9)

    def test_a_scenario_past_expiry_settles_at_intrinsic(self):
        """Rather than producing a negative-time artifact."""
        reaction = call(23_900, Scenario("expired", spot_change=200, days_elapsed=99))
        assert reaction.price_after == pytest.approx(214.0, abs=1.0)


class TestTheDecompositionAddsUp:
    def test_a_small_move_is_almost_all_delta(self):
        reaction = call(23_900, Scenario("small", spot_change=10.0))
        assert abs(reaction.gamma_contribution) < abs(reaction.delta_contribution) / 10

    def test_gamma_helps_the_buyer_in_both_directions(self):
        """Long gamma: the up-move gains more than the down-move loses, which
        is the asymmetry a buyer is paying theta for."""
        up = call(23_900, Scenario("up", spot_change=238.0))
        down = call(23_900, Scenario("down", spot_change=-238.0))
        assert up.change > abs(down.change)
        assert up.gamma_contribution > 0
        assert down.gamma_contribution > 0

    def test_the_estimate_tracks_the_exact_price_on_a_small_move(self):
        reaction = call(23_900, Scenario("small", spot_change=25.0))
        assert reaction.approximation_error == pytest.approx(0.0, abs=0.5)

    def test_the_error_grows_on_a_large_move(self):
        """Which is why both numbers are kept. A projection showing only the
        estimate would be most confident exactly where it is most wrong."""
        small = call(23_900, Scenario("small", spot_change=25.0))
        large = call(23_900, Scenario("large", spot_change=500.0))
        assert abs(large.approximation_error) > abs(small.approximation_error)

    def test_theta_is_charged_whichever_way_the_index_goes(self):
        """The cost buyers forget to price on an overnight hold."""
        for move in (-238.0, 0.0, 238.0):
            reaction = call(23_900, Scenario("gap", spot_change=move, days_elapsed=1.0))
            assert reaction.theta_contribution < 0

    def test_vega_only_moves_when_volatility_does(self):
        assert call(23_900, Scenario("m", spot_change=50)).vega_contribution == 0.0
        assert call(23_900, Scenario("v", iv_change=2.0)).vega_contribution > 0


class TestWhatABuyerNeedsToSee:
    def test_delta_rises_as_the_option_goes_in_the_money(self):
        reaction = call(23_900, Scenario("up", spot_change=238.0))
        assert reaction.delta_after > reaction.delta_before

    def test_a_flat_open_with_an_iv_crush_loses_money(self):
        """The most common way a directionally correct buyer still loses: the
        index does nothing and volatility reprices down, so the position pays
        through vega and theta at once."""
        reaction = call(23_900, volatility_crush())
        assert reaction.change < 0
        assert reaction.vega_contribution < 0
        assert reaction.theta_contribution < 0

    def test_the_rupee_figure_scales_with_the_lot(self):
        reaction = call(23_900, Scenario("up", spot_change=119.0))
        assert reaction.per_lot(65) == pytest.approx(reaction.change * 65)
        assert reaction.per_lot(65, lots=3) == pytest.approx(reaction.change * 195)

    def test_a_worthless_option_reports_no_percentage(self):
        """Rather than dividing by zero and reporting an infinite gain."""
        reaction = call(30_000, Scenario("flat", days_elapsed=1.0))
        assert reaction.price_before == pytest.approx(0.0, abs=1e-6)
        assert reaction.change_pct is None


class TestGapLadder:
    def test_it_is_expressed_in_sigmas_not_points(self):
        """150 points is a routine morning at 20% IV and a violent one at
        10%, so the ladder scales with the volatility that produced it."""
        quiet = gap_scenarios(119.0)
        wild = gap_scenarios(300.0)
        assert quiet[0].spot_change == pytest.approx(-238.0)
        assert wild[0].spot_change == pytest.approx(-600.0)

    def test_it_is_symmetric_around_a_flat_open(self):
        ladder = gap_scenarios(119.0)
        moves = [s.spot_change for s in ladder]
        assert moves == sorted(moves)
        assert 0.0 in moves
        assert moves[0] == -moves[-1]

    def test_an_overnight_gap_costs_a_day(self):
        """Always, whichever way the index goes."""
        assert all(s.days_elapsed == 1.0 for s in gap_scenarios(119.0))
        assert all(s.days_elapsed == 0.0 for s in gap_scenarios(119.0, overnight=False))


class TestTheDeltaFloor:
    """A rejection, not a score — and only on the leg that is the trade."""

    def test_the_default_floor_is_thirty(self):
        from index_option_brain.brain.config import StrikeEngineConfig

        assert StrikeEngineConfig().min_long_leg_delta == 0.30

    def test_a_credit_spread_wing_is_exempt(self, view):
        """Requiring 0.30 on a protective wing would make every defined-risk
        credit spread unbuildable, leaving only naked shorts — the opposite
        of safer."""
        from index_option_brain.brain.structures import build_structure
        from index_option_brain.contracts.enums import OrderSide, StrategyType

        candidate = build_structure(
            StrategyType.PUT_CREDIT_SPREAD, view, anchor_offset=-1, width_steps=1
        )
        assert candidate is not None
        assert candidate.is_credit
        long_legs = [leg for leg in candidate.legs if leg.side is OrderSide.BUY]
        assert long_legs, "a defined-risk credit spread must have a bought wing"
