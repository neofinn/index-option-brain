"""Quote-level pricing helpers.

`mid` and `relative_spread` are what the execution and liquidity paths price
from, so their edge cases matter: a stale LTP on an illiquid strike is a
classic source of phantom edge.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from index_option_brain.contracts.enums import OptionType
from index_option_brain.contracts.instruments import OptionContractSpec, OptionQuote

CONTRACT = OptionContractSpec(
    underlying_symbol="NIFTY",
    expiry=date(2026, 9, 10),
    strike=Decimal(24500),
    option_type=OptionType.CE,
    lot_size=75,
    tick_size=Decimal("0.05"),
)


def quote(bid: str | None, ask: str | None, ltp: str = "100") -> OptionQuote:
    return OptionQuote(
        contract=CONTRACT,
        timestamp=datetime(2026, 9, 4, 6, 0, tzinfo=UTC),
        ltp=Decimal(ltp),
        bid=None if bid is None else Decimal(bid),
        ask=None if ask is None else Decimal(ask),
        volume=1000,
        open_interest=10_000,
        open_interest_change=100,
        implied_volatility=Decimal(14),
    )


class TestMid:
    def test_mid_is_the_average_of_a_two_sided_quote(self):
        assert quote("98", "102").mid == Decimal(100)

    def test_mid_falls_back_to_ltp_when_a_side_is_missing(self):
        assert quote(None, "102", ltp="99").mid == Decimal(99)
        assert quote("98", None, ltp="99").mid == Decimal(99)

    def test_a_crossed_quote_falls_back_to_ltp(self):
        """An ask below the bid is bad data, not a negative spread."""
        assert quote("102", "98", ltp="99").mid == Decimal(99)


class TestSpread:
    def test_spread_and_relative_spread(self):
        subject = quote("98", "102")
        assert subject.spread == Decimal(4)
        assert subject.relative_spread == pytest.approx(Decimal("0.04"))

    def test_spread_is_none_without_both_sides(self):
        assert quote(None, "102").spread is None
        assert quote(None, "102").relative_spread is None

    def test_relative_spread_scales_with_premium(self):
        """The same absolute spread is far more punishing on a cheap option —
        which is why liquidity is measured relatively."""
        expensive = quote("298", "302", ltp="300").relative_spread
        cheap = quote("3", "7", ltp="5").relative_spread
        assert expensive is not None and cheap is not None
        assert cheap > expensive

    def test_relative_spread_of_a_worthless_quote_is_none(self):
        assert quote("0", "0", ltp="0").relative_spread is None
