"""Portfolio greeks.

The two properties worth most here: units cannot be confused silently, and
gamma is accounted for before a limit is judged. Both have already caused
real errors in this codebase or are one keystroke from doing so.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from index_option_brain.analytics.exposure import (
    leg_exposure,
    portfolio_exposure,
)
from index_option_brain.contracts.enums import OptionType, OrderSide
from index_option_brain.contracts.instruments import Greeks, OptionContractSpec

NIFTY_LOT = 65
BANKNIFTY_LOT = 30


def contract(
    symbol: str = "NIFTY", strike: float = 24100, lot: int = NIFTY_LOT
) -> OptionContractSpec:
    from datetime import date

    return OptionContractSpec(
        underlying_symbol=symbol,
        expiry=date(2026, 9, 8),
        strike=Decimal(str(strike)),
        option_type=OptionType.CE,
        lot_size=lot,
        tick_size=Decimal("0.05"),
    )


def greeks(delta: float, gamma: float = 0.0, theta: float = 0.0, vega: float = 0.0):
    return Greeks(
        delta=Decimal(str(delta)),
        gamma=Decimal(str(gamma)),
        theta=Decimal(str(theta)),
        vega=Decimal(str(vega)),
    )


class TestUnitsCannotBeGuessed:
    """PositionLeg.quantity is units; StrikeLeg.lots is lots. Reading one as
    the other is a 65x error on NIFTY that produces a plausible number."""

    def test_passing_neither_is_refused(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            leg_exposure(
                contract=contract(), side=OrderSide.BUY, greeks=greeks(0.45)
            )

    def test_passing_both_is_refused(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            leg_exposure(
                contract=contract(),
                side=OrderSide.BUY,
                greeks=greeks(0.45),
                lots=1,
                units=65,
            )

    def test_the_error_names_the_size_of_the_mistake(self) -> None:
        with pytest.raises(ValueError, match=r"65x error on NIFTY"):
            leg_exposure(contract=contract(), side=OrderSide.BUY, greeks=greeks(0.45))

    def test_lots_and_units_agree_when_converted_correctly(self) -> None:
        by_lots = leg_exposure(
            contract=contract(), side=OrderSide.BUY, greeks=greeks(0.45), lots=3
        )
        by_units = leg_exposure(
            contract=contract(),
            side=OrderSide.BUY,
            greeks=greeks(0.45),
            units=3 * NIFTY_LOT,
        )
        assert by_lots.units == by_units.units == 195
        assert by_lots.delta_units == by_units.delta_units


class TestSigns:
    def test_a_bought_leg_is_long_delta_and_a_sold_leg_is_short(self) -> None:
        long = leg_exposure(
            contract=contract(), side=OrderSide.BUY, greeks=greeks(0.45), lots=1
        )
        short = leg_exposure(
            contract=contract(), side=OrderSide.SELL, greeks=greeks(0.45), lots=1
        )
        assert long.delta_units == pytest.approx(29.25)
        assert short.delta_units == pytest.approx(-29.25)

    def test_every_greek_carries_the_side(self) -> None:
        """A sold option is short gamma and long theta — the sign has to
        follow through, or a short book reads as bleeding when it earns."""
        short = leg_exposure(
            contract=contract(),
            side=OrderSide.SELL,
            greeks=greeks(0.45, gamma=0.0015, theta=-8.0, vega=11.0),
            lots=1,
        )
        assert short.gamma_units is not None and short.gamma_units < 0
        assert short.theta_rupees is not None and short.theta_rupees > 0
        assert short.vega_rupees is not None and short.vega_rupees < 0


class TestGammaChangesExposure:
    """The reason a limit checked at entry is checked at the one moment it
    is guaranteed to pass."""

    def test_a_long_calls_delta_grows_into_a_rally(self) -> None:
        """A 0.383-delta call becomes about 0.533 delta on a 100-point
        move — 39% more exposure that nobody placed."""
        leg = leg_exposure(
            contract=contract(),
            side=OrderSide.BUY,
            greeks=greeks(0.383, gamma=0.00147),
            lots=1,
        )
        book = portfolio_exposure([(leg, Decimal(24000))])
        nifty = book.by_underlying["NIFTY"]

        assert nifty.delta_units == pytest.approx(24.9, rel=0.01)
        projected = nifty.projected_delta_units(100)
        assert projected == pytest.approx(34.4, rel=0.02)
        assert nifty.delta_growth(100) == pytest.approx(0.38, rel=0.05)

    def test_a_short_call_gets_shorter_into_a_rally(self) -> None:
        leg = leg_exposure(
            contract=contract(),
            side=OrderSide.SELL,
            greeks=greeks(0.383, gamma=0.00147),
            lots=1,
        )
        book = portfolio_exposure([(leg, Decimal(24000))])
        nifty = book.by_underlying["NIFTY"]
        assert nifty.projected_delta_units(100) < nifty.delta_units

    def test_a_flat_book_has_no_growth_ratio(self) -> None:
        """None, not infinity: an already-neutral book should not trip a
        limit expressed as a ratio."""
        long = leg_exposure(
            contract=contract(), side=OrderSide.BUY, greeks=greeks(0.5), lots=1
        )
        short = leg_exposure(
            contract=contract(), side=OrderSide.SELL, greeks=greeks(0.5), lots=1
        )
        book = portfolio_exposure([(long, Decimal(24000)), (short, Decimal(24000))])
        assert book.by_underlying["NIFTY"].delta_growth(100) is None


class TestUnderlyingsAreNotMixed:
    def test_deltas_are_kept_per_symbol(self) -> None:
        """NIFTY's lot is 65 around 24,000 and BANKNIFTY's is 30 around
        57,000. A sum of their deltas describes no portfolio."""
        nifty = leg_exposure(
            contract=contract("NIFTY", 24100, NIFTY_LOT),
            side=OrderSide.BUY,
            greeks=greeks(0.45),
            lots=1,
        )
        banknifty = leg_exposure(
            contract=contract("BANKNIFTY", 57500, BANKNIFTY_LOT),
            side=OrderSide.BUY,
            greeks=greeks(0.45),
            lots=1,
        )
        book = portfolio_exposure(
            [(nifty, Decimal(24000)), (banknifty, Decimal(57500))]
        )
        assert set(book.by_underlying) == {"NIFTY", "BANKNIFTY"}
        assert book.by_underlying["NIFTY"].delta_units == pytest.approx(29.25)
        assert book.by_underlying["BANKNIFTY"].delta_units == pytest.approx(13.5)

    def test_only_rupees_are_summed_across_symbols(self) -> None:
        nifty = leg_exposure(
            contract=contract("NIFTY", 24100, NIFTY_LOT),
            side=OrderSide.BUY,
            greeks=greeks(0.45),
            lots=1,
        )
        book = portfolio_exposure([(nifty, Decimal(24000))])
        assert book.delta_notional == pytest.approx(29.25 * 24000)

    def test_netting_and_gross_differ_and_both_are_reported(self) -> None:
        """Two opposing index positions net to little and can both lose;
        they are correlated, not identical, so netting is not a hedge."""
        long_n = leg_exposure(
            contract=contract("NIFTY", 24100, NIFTY_LOT),
            side=OrderSide.BUY,
            greeks=greeks(0.5),
            lots=1,
        )
        short_b = leg_exposure(
            contract=contract("BANKNIFTY", 57500, BANKNIFTY_LOT),
            side=OrderSide.SELL,
            greeks=greeks(0.5),
            lots=1,
        )
        book = portfolio_exposure(
            [(long_n, Decimal(24000)), (short_b, Decimal(57500))]
        )
        assert abs(book.delta_notional) < book.gross_delta_notional


class TestAbsence:
    def test_an_unmarked_leg_does_not_contribute_zero(self) -> None:
        """Zero delta is a hedged leg; unknown delta is an unmeasured one. A
        limit that passes because a leg was invisible is worse than none."""
        marked = leg_exposure(
            contract=contract(), side=OrderSide.BUY, greeks=greeks(0.45), lots=1
        )
        unmarked = leg_exposure(
            contract=contract(), side=OrderSide.BUY, greeks=None, lots=1
        )
        book = portfolio_exposure(
            [(marked, Decimal(24000)), (unmarked, Decimal(24000))]
        )
        nifty = book.by_underlying["NIFTY"]

        assert nifty.unmeasured_legs == 1
        assert nifty.is_complete is False
        assert book.is_complete is False
        # The total is the measured part only, and therefore a floor.
        assert nifty.delta_units == pytest.approx(29.25)

    def test_an_empty_book_is_not_complete(self) -> None:
        book = portfolio_exposure([])
        assert book.is_complete is False
        assert book.delta_notional == 0.0


class TestCapital:
    def test_exposure_is_a_multiple_of_capital_not_a_percentage(self) -> None:
        """Three lots of a 90-rupee call cost ~17,500 and carry over 20 lakh
        of delta notional. "1,200%" would read as a bug."""
        leg = leg_exposure(
            contract=contract(), side=OrderSide.BUY, greeks=greeks(0.45), lots=3
        )
        book = portfolio_exposure([(leg, Decimal(24000))])

        multiple = book.delta_share_of_capital(Decimal(500_000))
        assert multiple is not None
        assert multiple == pytest.approx(87.75 * 24000 / 500_000, rel=1e-6)
        assert multiple > 4

    def test_no_capital_yields_none_rather_than_a_division(self) -> None:
        leg = leg_exposure(
            contract=contract(), side=OrderSide.BUY, greeks=greeks(0.45), lots=1
        )
        book = portfolio_exposure([(leg, Decimal(24000))])
        assert book.delta_share_of_capital(Decimal(0)) is None


class TestProjectionAcrossUnderlyings:
    def test_each_symbol_moves_by_its_own_sigma(self) -> None:
        """100 points is a different event on NIFTY than on BANKNIFTY."""
        nifty = leg_exposure(
            contract=contract("NIFTY", 24100, NIFTY_LOT),
            side=OrderSide.BUY,
            greeks=greeks(0.4, gamma=0.0015),
            lots=1,
        )
        banknifty = leg_exposure(
            contract=contract("BANKNIFTY", 57500, BANKNIFTY_LOT),
            side=OrderSide.BUY,
            greeks=greeks(0.4, gamma=0.0008),
            lots=1,
        )
        book = portfolio_exposure(
            [(nifty, Decimal(24000)), (banknifty, Decimal(57500))]
        )
        projected = book.projected_delta_notional(
            1.0, {"NIFTY": 140.0, "BANKNIFTY": 400.0}
        )
        assert projected > book.delta_notional

    def test_a_symbol_with_no_sigma_is_projected_unchanged(self) -> None:
        """Rather than moved by a sigma borrowed from a different index."""
        leg = leg_exposure(
            contract=contract(), side=OrderSide.BUY, greeks=greeks(0.4, gamma=0.0015), lots=1
        )
        book = portfolio_exposure([(leg, Decimal(24000))])
        assert book.projected_delta_notional(1.0, {}) == pytest.approx(
            book.delta_notional
        )
