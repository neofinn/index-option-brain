"""Tests for the Black-Scholes layer.

Greeks are computed here because no Indian feed publishes them, which makes
this module load-bearing for the Strike Engine's delta-fit ranking. So the
assertions below are mostly *properties* — parity, bounds, monotonicity,
limits — rather than golden numbers copied out of the implementation, since a
golden number only proves the code still does what it did yesterday.
"""

from __future__ import annotations

import math

import pytest

from index_option_brain.analytics.pricing import (
    DEFAULT_RISK_FREE_RATE,
    greeks_from_iv,
    implied_volatility,
    price_option,
)
from index_option_brain.contracts.enums import OptionType

# A realistic Indian weekly: NIFTY near 23,900 with six calendar days to a
# Tuesday expiry and IV in the low teens.
SPOT = 23_900.0
WEEK = 6 / 365
IV = 0.12


def call(strike: float, *, years: float = WEEK, iv: float = IV):
    return price_option(
        spot=SPOT, strike=strike, years=years, iv=iv, option_type=OptionType.CE
    )


def put(strike: float, *, years: float = WEEK, iv: float = IV):
    return price_option(
        spot=SPOT, strike=strike, years=years, iv=iv, option_type=OptionType.PE
    )


class TestPutCallParity:
    @pytest.mark.parametrize("strike", [22_000.0, 23_900.0, 25_000.0])
    def test_parity_holds(self, strike: float):
        """C - P == S - K*exp(-rT). Parity is the single strongest check on a
        pricer: it is arbitrage, not a modelling choice, so a violation means
        the code is wrong regardless of how plausible the premiums look."""
        discounted = strike * math.exp(-DEFAULT_RISK_FREE_RATE * WEEK)
        assert call(strike).price - put(strike).price == pytest.approx(
            SPOT - discounted, abs=1e-6
        )

    @pytest.mark.parametrize("strike", [22_000.0, 23_900.0, 25_000.0])
    def test_delta_parity_holds(self, strike: float):
        """Call delta minus put delta is exactly 1 for European options."""
        assert call(strike).delta - put(strike).delta == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize("strike", [22_000.0, 23_900.0, 25_000.0])
    def test_gamma_and_vega_are_shared(self, strike: float):
        """A call and a put on the same strike share gamma and vega. This is
        what lets a spread's net gamma be computed by adding legs."""
        assert call(strike).gamma == pytest.approx(put(strike).gamma, rel=1e-12)
        assert call(strike).vega == pytest.approx(put(strike).vega, rel=1e-12)


class TestPriceBounds:
    def test_price_never_below_the_european_lower_bound(self):
        """The floor is the *discounted* strike, not the strike. A European
        put can legitimately trade below its cash intrinsic value because the
        strike is only receivable at expiry — the same fact that makes a
        deep-ITM put's theta positive. Asserting the American bound here
        would be asserting the wrong model."""
        for strike in (20_000.0, 23_000.0, 23_900.0, 24_800.0, 28_000.0):
            discounted = strike * math.exp(-DEFAULT_RISK_FREE_RATE * WEEK)
            assert call(strike).price >= max(0.0, SPOT - discounted) - 1e-6
            assert put(strike).price >= max(0.0, discounted - SPOT) - 1e-6
            assert call(strike).price >= 0.0
            assert put(strike).price >= 0.0

    def test_call_price_never_above_spot(self):
        assert call(1.0).price < SPOT

    def test_deep_out_of_the_money_is_nearly_worthless(self):
        """A 28,000 call six days out on 12% IV is a lottery ticket, and the
        pricer has to say so rather than returning a tradeable premium."""
        assert call(28_000.0).price < 1.0

    def test_price_increases_with_strike_for_puts(self):
        prices = [put(k).price for k in (23_000.0, 23_500.0, 23_900.0, 24_300.0)]
        assert prices == sorted(prices)

    def test_price_decreases_with_strike_for_calls(self):
        prices = [call(k).price for k in (23_000.0, 23_500.0, 23_900.0, 24_300.0)]
        assert prices == sorted(prices, reverse=True)

    def test_price_increases_with_volatility(self):
        prices = [call(23_900.0, iv=iv).price for iv in (0.05, 0.10, 0.20, 0.40)]
        assert prices == sorted(prices)

    def test_price_increases_with_time(self):
        prices = [call(23_900.0, years=y).price for y in (1 / 365, WEEK, 30 / 365)]
        assert prices == sorted(prices)


class TestDelta:
    def test_call_delta_within_zero_and_one(self):
        for strike in (20_000.0, 23_900.0, 28_000.0):
            assert 0.0 <= call(strike).delta <= 1.0

    def test_put_delta_within_minus_one_and_zero(self):
        for strike in (20_000.0, 23_900.0, 28_000.0):
            assert -1.0 <= put(strike).delta <= 0.0

    def test_at_the_money_delta_is_near_half(self):
        """The delta-fit ranking in the Strike Engine keys off this: a strike
        the chain calls ATM must come back near 0.5, or every ranked
        selection is shifted."""
        assert call(23_900.0).delta == pytest.approx(0.5, abs=0.05)
        assert put(23_900.0).delta == pytest.approx(-0.5, abs=0.05)

    def test_deep_in_the_money_call_delta_approaches_one(self):
        assert call(15_000.0).delta == pytest.approx(1.0, abs=1e-6)

    def test_deep_out_of_the_money_call_delta_approaches_zero(self):
        assert call(35_000.0).delta == pytest.approx(0.0, abs=1e-6)

    def test_delta_is_monotonic_in_strike(self):
        deltas = [call(k).delta for k in (22_000.0, 23_000.0, 23_900.0, 25_000.0)]
        assert deltas == sorted(deltas, reverse=True)


class TestGammaAndVega:
    def test_gamma_is_positive(self):
        assert call(23_900.0).gamma > 0

    def test_gamma_peaks_near_the_money(self):
        """Gamma concentration at ATM is why short-gamma structures are
        dangerous near expiry, and the Position brain reads it."""
        atm = call(23_900.0).gamma
        assert atm > call(22_500.0).gamma
        assert atm > call(25_300.0).gamma

    def test_gamma_rises_as_expiry_approaches(self):
        assert call(23_900.0, years=1 / 365).gamma > call(23_900.0, years=30 / 365).gamma

    def test_vega_is_positive(self):
        assert call(23_900.0).vega > 0

    def test_vega_is_quoted_per_iv_point(self):
        """Vega must predict the price move for a 1-point IV change, because
        that is the unit the volatility brain reasons in. A vega quoted per
        unit of decimal IV would be 100x too large and would silently make
        every IV-driven adjustment absurd."""
        base = call(23_900.0, iv=0.12)
        bumped = call(23_900.0, iv=0.13)
        assert bumped.price - base.price == pytest.approx(base.vega, rel=0.02)

    def test_vega_falls_as_expiry_approaches(self):
        assert call(23_900.0, years=1 / 365).vega < call(23_900.0, years=30 / 365).vega


class TestTheta:
    def test_theta_is_quoted_per_calendar_day(self):
        """An option held one more day loses roughly theta. Calendar days
        rather than trading days matter on a Tuesday-expiry weekly, where a
        position carried over a weekend decays for three days and a
        trading-day convention would understate it by two."""
        today = call(23_900.0, years=WEEK)
        tomorrow = call(23_900.0, years=WEEK - 1 / 365)
        assert tomorrow.price - today.price == pytest.approx(today.theta, rel=0.05)

    def test_at_the_money_theta_is_negative(self):
        assert call(23_900.0).theta < 0
        assert put(23_900.0).theta < 0

    def test_deep_in_the_money_put_theta_is_positive(self):
        """Not a bug. A deep-ITM European put is worth less than its
        intrinsic value because the strike is only receivable at expiry, so
        it *gains* as expiry approaches. Asserting theta < 0 everywhere would
        be asserting a wrong model."""
        assert put(30_000.0).theta > 0

    def test_weekend_costs_three_days_of_decay(self):
        friday = call(23_900.0, years=10 / 365)
        monday = call(23_900.0, years=7 / 365)
        assert friday.price - monday.price == pytest.approx(
            -3 * friday.theta, rel=0.10
        )


class TestDegenerateCases:
    def test_at_expiry_the_option_is_worth_intrinsic(self):
        expired = price_option(
            spot=SPOT, strike=23_000.0, years=0.0, iv=IV, option_type=OptionType.CE
        )
        assert expired.price == pytest.approx(900.0)
        assert expired.is_degenerate

    def test_expired_out_of_the_money_is_worthless(self):
        expired = price_option(
            spot=SPOT, strike=24_500.0, years=0.0, iv=IV, option_type=OptionType.CE
        )
        assert expired.price == 0.0
        assert expired.delta == 0.0

    def test_expired_in_the_money_delta_is_one(self):
        """After expiry an ITM option is a fixed cash claim, so it tracks the
        index one-for-one and its sensitivities are gone."""
        expired = price_option(
            spot=SPOT, strike=23_000.0, years=0.0, iv=IV, option_type=OptionType.CE
        )
        assert expired.delta == 1.0
        assert expired.gamma == 0.0
        assert expired.theta == 0.0
        assert expired.vega == 0.0

    def test_expired_in_the_money_put_delta_is_minus_one(self):
        expired = price_option(
            spot=SPOT, strike=24_500.0, years=0.0, iv=IV, option_type=OptionType.PE
        )
        assert expired.delta == -1.0

    def test_zero_volatility_collapses_to_intrinsic(self):
        frozen = price_option(
            spot=SPOT, strike=23_000.0, years=WEEK, iv=0.0, option_type=OptionType.CE
        )
        assert frozen.price == pytest.approx(900.0)
        assert frozen.is_degenerate

    def test_a_live_option_is_not_degenerate(self):
        assert not call(23_900.0).is_degenerate


class TestPercentWrapper:
    def test_iv_percent_matches_decimal_iv(self):
        """The live NSE chain reports IV as 11.43, not 0.1143. Getting this
        conversion wrong by 100x would not crash — it would quietly price
        every option as if volatility were 1,143%."""
        from_percent = greeks_from_iv(
            spot=SPOT,
            strike=23_900.0,
            years=WEEK,
            iv_percent=11.43,
            option_type=OptionType.CE,
        )
        from_decimal = call(23_900.0, iv=0.1143)
        assert from_percent.price == pytest.approx(from_decimal.price)
        assert from_percent.delta == pytest.approx(from_decimal.delta)

    def test_a_realistic_live_premium_comes_back(self):
        """NSE on 02-Sep-2026: NIFTY 23,914.45, the 23,900 CE expiring
        08-Sep-2026 marked 131.60 at 8.39 IV. Six calendar days of 8.39 vol
        should produce a premium in that neighbourhood — not a proof of the
        market's price, but a check that the whole unit chain (percent IV,
        calendar years, index points) lines up on real inputs."""
        result = greeks_from_iv(
            spot=23_914.45,
            strike=23_900.0,
            years=6 / 365,
            iv_percent=8.39,
            option_type=OptionType.CE,
        )
        assert 90.0 < result.price < 180.0
        assert 0.45 < result.delta < 0.65


class TestImpliedVolatility:
    @pytest.mark.parametrize("iv", [0.06, 0.12, 0.35, 0.80])
    @pytest.mark.parametrize("strike", [23_000.0, 23_900.0, 24_800.0])
    def test_round_trip_recovers_the_input(self, iv: float, strike: float):
        price = call(strike, iv=iv).price
        recovered = implied_volatility(
            market_price=price,
            spot=SPOT,
            strike=strike,
            years=WEEK,
            option_type=OptionType.CE,
        )
        assert recovered is not None
        assert recovered == pytest.approx(iv, abs=1e-4)

    def test_round_trip_works_for_puts(self):
        price = put(23_900.0, iv=0.18).price
        recovered = implied_volatility(
            market_price=price,
            spot=SPOT,
            strike=23_900.0,
            years=WEEK,
            option_type=OptionType.PE,
        )
        assert recovered is not None
        assert recovered == pytest.approx(0.18, abs=1e-4)

    def test_below_intrinsic_returns_none(self):
        """A premium under intrinsic value is stale or crossed data. There is
        no volatility that explains it, so the honest answer is None rather
        than a floor value that would look like a real reading."""
        assert (
            implied_volatility(
                market_price=100.0,
                spot=SPOT,
                strike=23_000.0,
                years=WEEK,
                option_type=OptionType.CE,
            )
            is None
        )

    def test_absurdly_high_price_returns_none(self):
        assert (
            implied_volatility(
                market_price=SPOT * 0.99,
                spot=SPOT,
                strike=23_900.0,
                years=WEEK,
                option_type=OptionType.CE,
            )
            is None
        )

    def test_zero_price_returns_none(self):
        assert (
            implied_volatility(
                market_price=0.0,
                spot=SPOT,
                strike=23_900.0,
                years=WEEK,
                option_type=OptionType.CE,
            )
            is None
        )

    def test_expired_option_returns_none(self):
        assert (
            implied_volatility(
                market_price=50.0,
                spot=SPOT,
                strike=23_900.0,
                years=0.0,
                option_type=OptionType.CE,
            )
            is None
        )

    def test_recovered_iv_reprices_to_the_input_premium(self):
        """The property that actually matters downstream: whatever IV comes
        back must reprice the option to the premium it was derived from."""
        premium = 131.60
        iv = implied_volatility(
            market_price=premium,
            spot=23_914.45,
            strike=23_900.0,
            years=6 / 365,
            option_type=OptionType.CE,
        )
        assert iv is not None
        repriced = price_option(
            spot=23_914.45,
            strike=23_900.0,
            years=6 / 365,
            iv=iv,
            option_type=OptionType.CE,
        )
        assert repriced.price == pytest.approx(premium, abs=1e-4)
