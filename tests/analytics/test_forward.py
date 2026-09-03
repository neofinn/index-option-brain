"""Tests for the forward solve and the carry term it feeds.

The bug these pin: pricing a NIFTY option off `spot * exp(rate * years)`
puts the futures basis into the volatility instead. On 3 Sep 2026 that made
the same strike solve to a 10.6% call IV and an 8.7% put IV, and moved the
24,200 call's delta from 0.304 to 0.290 — across the 0.30 floor that decides
whether the strike is buyable at all.
"""

from __future__ import annotations

import math

import pytest

from index_option_brain.analytics.pricing import (
    DEFAULT_RISK_FREE_RATE,
    forward_from_parity,
    implied_volatility,
    price_option,
)
from index_option_brain.contracts.enums import OptionType

YEARS = 5.24 / 365.0


def synthetic_pairs(forward: float, iv: float, strikes: list[float]) -> list[tuple[float, float, float]]:
    """Calls and puts priced off a known forward, for round-tripping."""
    carry = DEFAULT_RISK_FREE_RATE - math.log(forward / 24000.0) / YEARS
    out = []
    for k in strikes:
        common: dict[str, float] = {
            "spot": 24000.0,
            "strike": k,
            "years": YEARS,
            "iv": iv,
            "rate": DEFAULT_RISK_FREE_RATE,
            "dividend_yield": carry,
        }
        out.append(
            (
                k,
                price_option(option_type=OptionType.CE, **common).price,
                price_option(option_type=OptionType.PE, **common).price,
            )
        )
    return out


class TestCarry:
    def test_zero_carry_reproduces_plain_black_scholes(self) -> None:
        """The default must not move a single existing number."""
        common = {
            "spot": 24000.0,
            "strike": 24100.0,
            "years": YEARS,
            "iv": 0.10,
            "option_type": OptionType.CE,
        }
        assert price_option(**common).price == price_option(
            **common, dividend_yield=0.0
        ).price

    def test_put_call_parity_holds_under_carry(self) -> None:
        """C - P == (F - K) * exp(-rT). If this fails the carry term is wrong
        and every IV solved through it inherits the error."""
        forward, carry = 24064.0, 0.0
        carry = DEFAULT_RISK_FREE_RATE - math.log(forward / 24000.0) / YEARS
        common = {
            "spot": 24000.0,
            "strike": 24050.0,
            "years": YEARS,
            "iv": 0.0966,
            "rate": DEFAULT_RISK_FREE_RATE,
            "dividend_yield": carry,
        }
        call = price_option(option_type=OptionType.CE, **common).price
        put = price_option(option_type=OptionType.PE, **common).price
        expected = (forward - 24050.0) * math.exp(-DEFAULT_RISK_FREE_RATE * YEARS)
        assert call - put == pytest.approx(expected, abs=1e-6)

    def test_a_call_and_a_put_solve_to_the_same_iv_under_the_right_forward(
        self,
    ) -> None:
        """The observed symptom, inverted into a test.

        Off spot these disagreed by nearly 2 IV points at the same strike.
        Off the forward the market actually quotes they must agree.
        """
        forward, true_iv = 24064.14, 0.0966
        pairs = synthetic_pairs(forward, true_iv, [24050.0])
        _, call, put = pairs[0]
        carry = DEFAULT_RISK_FREE_RATE - math.log(forward / 24000.0) / YEARS

        call_iv = implied_volatility(
            market_price=call,
            spot=24000.0,
            strike=24050.0,
            years=YEARS,
            option_type=OptionType.CE,
            dividend_yield=carry,
        )
        put_iv = implied_volatility(
            market_price=put,
            spot=24000.0,
            strike=24050.0,
            years=YEARS,
            option_type=OptionType.PE,
            dividend_yield=carry,
        )
        assert call_iv == pytest.approx(put_iv, abs=1e-4)
        assert call_iv == pytest.approx(true_iv, abs=1e-4)

    def test_ignoring_a_real_basis_biases_the_delta_across_the_floor(self) -> None:
        """Why this is not a rounding concern: the 24,200 call on 3 Sep 2026."""
        forward = 24064.14
        carry = DEFAULT_RISK_FREE_RATE - math.log(forward / 24020.4) / YEARS
        common = {
            "spot": 24020.4,
            "strike": 24200.0,
            "years": YEARS,
            "iv": 0.0907,
            "option_type": OptionType.CE,
        }
        off_spot = price_option(**common).delta
        off_forward = price_option(**common, dividend_yield=carry).delta

        assert off_spot < 0.30 < off_forward


class TestForwardFromParity:
    def test_it_recovers_the_forward_it_was_priced_from(self) -> None:
        pairs = synthetic_pairs(24064.14, 0.0966, [23950.0, 24000.0, 24050.0, 24100.0])
        estimate = forward_from_parity(pairs=pairs, spot=24000.0, years=YEARS)

        assert estimate is not None
        assert estimate.forward == pytest.approx(24064.14, abs=0.01)
        assert estimate.strikes_used == 4

    def test_it_separates_carry_from_positioning(self) -> None:
        """Pure carry is mechanical and says nothing. The excess is the signal."""
        pairs = synthetic_pairs(24064.14, 0.0966, [24000.0, 24050.0])
        estimate = forward_from_parity(pairs=pairs, spot=24000.0, years=YEARS)

        assert estimate is not None
        assert estimate.basis == pytest.approx(64.14, abs=0.01)
        assert estimate.carry_basis == pytest.approx(22.4, abs=0.5)
        assert estimate.excess_basis == pytest.approx(estimate.basis - estimate.carry_basis)

    def test_the_recovered_carry_reprices_to_the_same_forward(self) -> None:
        pairs = synthetic_pairs(24064.14, 0.0966, [24000.0, 24050.0])
        estimate = forward_from_parity(pairs=pairs, spot=24000.0, years=YEARS)

        assert estimate is not None
        implied = 24000.0 * math.exp(
            (DEFAULT_RISK_FREE_RATE - estimate.dividend_yield) * YEARS
        )
        assert implied == pytest.approx(estimate.forward, abs=0.01)

    def test_nothing_usable_yields_none_not_a_number(self) -> None:
        """An invented forward would silently reprice the whole chain."""
        assert forward_from_parity(pairs=[], spot=24000.0, years=YEARS) is None
        assert (
            forward_from_parity(
                pairs=[(24000.0, 0.0, 120.0)], spot=24000.0, years=YEARS
            )
            is None
        )
        assert (
            forward_from_parity(pairs=[(24000.0, 100.0, 120.0)], spot=24000.0, years=0.0)
            is None
        )

    def test_it_prefers_the_strikes_nearest_the_money(self) -> None:
        """Where both legs are liquid and the parity difference is least noisy.

        A far strike is given a deliberately corrupt quote; the estimate must
        not move, because it should never have been consulted.
        """
        good = synthetic_pairs(24064.14, 0.0966, [24000.0, 24050.0, 23950.0])
        estimate = forward_from_parity(
            pairs=[*good, (30000.0, 1.0, 6000.0)],
            spot=24000.0,
            years=YEARS,
            max_strikes=3,
        )
        assert estimate is not None
        assert estimate.forward == pytest.approx(24064.14, abs=0.01)
