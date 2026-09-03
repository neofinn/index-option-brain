"""Gamma against theta.

The first class is the one that matters: the breakeven move a long option
needs is not something to calculate per position, it is the implied move,
identically. Everything after that is about the costs the identity omits.
"""

from __future__ import annotations

import math
from decimal import Decimal

import pytest

from index_option_brain.analytics.gamma import (
    GammaAssessment,
    breakeven_move,
    hedge_drag,
    implied_move,
)
from index_option_brain.analytics.pricing import price_option
from index_option_brain.contracts.enums import OptionType

SPOT = 24000.0
IV = 0.0966
NIFTY_LOT = 65


def atm(days: float):
    return price_option(
        spot=SPOT,
        strike=SPOT,
        years=days / 365.0,
        iv=IV,
        option_type=OptionType.CE,
        # Zero rate isolates the identity: the carry terms in theta are real
        # but they are not part of the gamma-theta trade-off.
        rate=0.0,
    )


class TestTheIdentity:
    @pytest.mark.parametrize("days", [1, 3, 7, 30, 90])
    def test_the_breakeven_move_is_the_implied_move(self, days: int) -> None:
        """Falls out of the Black-Scholes PDE, where
        theta = -0.5 * spot^2 * iv^2 * gamma. So "how far must NIFTY move for
        gamma to beat theta" has one answer for every option on the board."""
        greeks = atm(days)
        breakeven = breakeven_move(gamma=greeks.gamma, theta=greeks.theta, days=1.0)
        implied = implied_move(spot=SPOT, iv_percent=IV * 100, days=1.0)

        assert breakeven is not None
        assert breakeven / implied == pytest.approx(1.0, abs=1e-4)

    @pytest.mark.parametrize("strike", [23000.0, 24000.0, 25000.0])
    def test_it_holds_away_from_the_money_too(self, strike: float) -> None:
        """No strike is "better for gamma" — they all sit at the same bar."""
        greeks = price_option(
            spot=SPOT, strike=strike, years=7 / 365, iv=IV,
            option_type=OptionType.CE, rate=0.0,
        )
        breakeven = breakeven_move(gamma=greeks.gamma, theta=greeks.theta)
        implied = implied_move(spot=SPOT, iv_percent=IV * 100)
        assert breakeven is not None
        assert breakeven / implied == pytest.approx(1.0, abs=1e-3)

    def test_calendar_days_not_trading_days(self) -> None:
        """Theta decays over the weekend. Using 252 here would break the
        identity by about 25% and look like a modelling error."""
        greeks = atm(7)
        breakeven = breakeven_move(gamma=greeks.gamma, theta=greeks.theta)
        assert breakeven is not None
        wrong = SPOT * IV * math.sqrt(1 / 252)
        assert breakeven != pytest.approx(wrong, rel=0.05)


class TestAbsence:
    def test_no_convexity_has_no_breakeven(self) -> None:
        """A position with no gamma has no move that rescues it, which is not
        a breakeven of zero."""
        assert breakeven_move(gamma=0.0, theta=-10.0) is None
        assert breakeven_move(gamma=0.001, theta=-10.0, days=0) is None

    def test_an_unmeasured_tape_yields_no_verdict(self) -> None:
        """An unmeasured tape is not a calm one."""
        greeks = atm(7)
        assessment = GammaAssessment(
            spot=SPOT, gamma=greeks.gamma, theta=greeks.theta,
            iv_percent=IV * 100, realized_percent=None,
        )
        assert assessment.realized_daily_move is None
        assert assessment.pays is None
        assert "unmeasured" in assessment.describe()


class TestCosts:
    """The term the closed form omits, and the one that decides outcomes."""

    def _drag(self, hedges: int):
        return hedge_drag(
            hedges_per_day=hedges,
            hedge_notional=Decimal(50_000),
            spread_fraction=0.0005,
        )

    def test_hedging_raises_the_bar_above_the_implied_move(self) -> None:
        greeks = atm(7)
        assessment = GammaAssessment(
            spot=SPOT, gamma=greeks.gamma, theta=greeks.theta,
            iv_percent=IV * 100, realized_percent=IV * 100,
            units=NIFTY_LOT, drag=self._drag(4),
        )
        assert assessment.breakeven_with_costs is not None
        assert assessment.frictionless_breakeven is not None
        assert assessment.breakeven_with_costs > assessment.frictionless_breakeven
        assert (assessment.cost_penalty or 0) > 0

    def test_hedging_more_often_costs_linearly_more(self) -> None:
        """So the optimum has nothing to do with the greeks."""
        assert self._drag(8).daily_cost == self._drag(4).daily_cost * 2

    def test_costs_are_compared_per_unit_not_to_the_whole_position(self) -> None:
        """The bug this signature exists to prevent: theta is per unit of the
        underlying and daily_cost is rupees for the position, so adding them
        directly inflates the breakeven by the lot size — 176 points became
        1,038, which reads as "gamma never pays" rather than as an error."""
        greeks = atm(7)
        drag = self._drag(4)
        one_lot = drag.move_needed(
            gamma=greeks.gamma, theta=greeks.theta, units=NIFTY_LOT
        )
        assert one_lot is not None
        assert 150 < one_lot < 250

        # More units spread the same hedging bill further, so the bar falls
        # toward the frictionless one rather than rising.
        ten_lots = drag.move_needed(
            gamma=greeks.gamma, theta=greeks.theta, units=NIFTY_LOT * 10
        )
        assert ten_lots is not None
        assert ten_lots < one_lot

    def test_a_sizeless_position_cannot_be_costed(self) -> None:
        greeks = atm(7)
        assert self._drag(4).move_needed(
            gamma=greeks.gamma, theta=greeks.theta, units=0
        ) is None

    def test_realized_equal_to_implied_loses_once_costs_are_counted(self) -> None:
        """A mathematically fair trade is a losing one after friction — the
        whole reason the identity is not the end of the analysis."""
        greeks = atm(7)
        assessment = GammaAssessment(
            spot=SPOT, gamma=greeks.gamma, theta=greeks.theta,
            iv_percent=IV * 100, realized_percent=IV * 100,
            units=NIFTY_LOT, drag=self._drag(4),
        )
        assert assessment.pays is False

    def test_a_tape_delivering_well_above_implied_clears_the_bar(self) -> None:
        greeks = atm(7)
        assessment = GammaAssessment(
            spot=SPOT, gamma=greeks.gamma, theta=greeks.theta,
            iv_percent=IV * 100, realized_percent=IV * 100 * 2.5,
            units=NIFTY_LOT, drag=self._drag(4),
        )
        assert assessment.pays is True
        assert "clears it" in assessment.describe()

    def test_not_hedging_at_all_leaves_the_bar_frictionless(self) -> None:
        greeks = atm(7)
        assessment = GammaAssessment(
            spot=SPOT, gamma=greeks.gamma, theta=greeks.theta,
            iv_percent=IV * 100, realized_percent=IV * 100,
        )
        assert assessment.breakeven_with_costs == assessment.frictionless_breakeven


class TestDescription:
    def test_it_names_the_identity_rather_than_implying_a_calculation(self) -> None:
        greeks = atm(7)
        text = GammaAssessment(
            spot=SPOT, gamma=greeks.gamma, theta=greeks.theta,
            iv_percent=IV * 100, realized_percent=IV * 100,
        ).describe()
        assert "identically" in text
