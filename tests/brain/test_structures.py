"""Option structure economics, verified arithmetically.

This module is where a mistake would be most expensive and least visible: the
Risk Engine authorizes against `max_loss`, so an off-by-a-multiplier here
becomes a position sized on a number that was never true. Every case below
uses a hand-built chain with exact prices so the expected values are computed
by hand, not by re-running the implementation.

Convention under test: buys price at the ask, sells at the bid, and money
values are per-position totals (premium x lot size x lots).
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from index_option_brain.brain import structures
from index_option_brain.contracts.enums import OptionType, OrderSide, StrategyType
from index_option_brain.contracts.instruments import (
    Greeks,
    OptionContractSpec,
    OptionQuote,
)

LOT_SIZE = 75
EXPIRY = date(2026, 9, 10)

# strike -> option type -> (bid, ask)
PRICES: dict[int, dict[OptionType, tuple[str, str]]] = {
    24300: {OptionType.CE: ("245", "247"), OptionType.PE: ("40", "42")},
    24400: {OptionType.CE: ("170", "172"), OptionType.PE: ("55", "57")},
    24500: {OptionType.CE: ("100", "102"), OptionType.PE: ("98", "100")},
    24600: {OptionType.CE: ("60", "62"), OptionType.PE: ("155", "157")},
    24700: {OptionType.CE: ("32", "34"), OptionType.PE: ("240", "242")},
}


def _quote(strike: int, option_type: OptionType) -> OptionQuote:
    bid, ask = PRICES[strike][option_type]
    return OptionQuote(
        contract=OptionContractSpec(
            underlying_symbol="NIFTY",
            expiry=EXPIRY,
            strike=Decimal(strike),
            option_type=option_type,
            lot_size=LOT_SIZE,
            tick_size=Decimal("0.05"),
        ),
        timestamp=datetime(2026, 9, 4, 6, 0, tzinfo=UTC),
        ltp=(Decimal(bid) + Decimal(ask)) / 2,
        bid=Decimal(bid),
        ask=Decimal(ask),
        volume=100_000,
        open_interest=500_000,
        open_interest_change=10_000,
        implied_volatility=Decimal("14.0"),
        greeks=Greeks(
            delta=Decimal("0.5"),
            gamma=Decimal("0.0001"),
            theta=Decimal(-8),
            vega=Decimal(12),
        ),
    )


@pytest.fixture
def view() -> structures.ChainView:
    chain = [_quote(strike, kind) for strike in PRICES for kind in (OptionType.CE, OptionType.PE)]
    built = structures.ChainView.from_chain(chain, Decimal(24500))
    assert built is not None
    return built


class TestChainView:
    def test_infers_atm_strike_step_and_lot_size(self, view: structures.ChainView):
        assert view.atm_strike == Decimal(24500)
        assert view.step == Decimal(100)
        assert view.lot_size == LOT_SIZE

    def test_strike_offsets_are_relative_to_atm(self, view: structures.ChainView):
        assert view.strike_at(0) == Decimal(24500)
        assert view.strike_at(2) == Decimal(24700)
        assert view.strike_at(-2) == Decimal(24300)

    def test_offsets_beyond_the_chain_return_none(self, view: structures.ChainView):
        assert view.strike_at(50) is None
        assert view.strike_at(-50) is None

    def test_an_empty_chain_has_no_view(self):
        assert structures.ChainView.from_chain([], Decimal(24500)) is None


class TestExecutionPricing:
    def test_buys_pay_the_ask_and_sells_receive_the_bid(self):
        quote = _quote(24500, OptionType.CE)
        assert structures.execution_price(quote, OrderSide.BUY) == Decimal(102)
        assert structures.execution_price(quote, OrderSide.SELL) == Decimal(100)

    def test_pricing_is_pessimistic_relative_to_mid(self):
        """Mid-pricing flatters every downstream number, and the flattery
        compounds through the Risk Engine."""
        quote = _quote(24500, OptionType.CE)
        assert structures.execution_price(quote, OrderSide.BUY) > quote.mid
        assert structures.execution_price(quote, OrderSide.SELL) < quote.mid


class TestLongOptions:
    def test_long_call_economics(self, view: structures.ChainView):
        candidate = structures.build_structure(StrategyType.LONG_CALL, view, anchor_offset=0)
        assert candidate is not None
        # Buy 24500 CE at the 102 ask.
        assert candidate.net_premium == Decimal(102 * LOT_SIZE)
        assert candidate.max_loss == Decimal(102 * LOT_SIZE)
        assert candidate.max_profit is None, "a long call's upside is unbounded"
        assert candidate.breakeven == [Decimal(24602)]
        assert candidate.capital_required == Decimal(102 * LOT_SIZE)
        assert not candidate.is_credit

    def test_long_put_breakeven_is_below_the_strike(self, view: structures.ChainView):
        candidate = structures.build_structure(StrategyType.LONG_PUT, view, anchor_offset=0)
        assert candidate is not None
        # Buy 24500 PE at the 100 ask.
        assert candidate.breakeven == [Decimal(24400)]
        assert candidate.max_loss == Decimal(100 * LOT_SIZE)

    def test_net_delta_scales_by_lot_size(self, view: structures.ChainView):
        candidate = structures.build_structure(StrategyType.LONG_CALL, view, anchor_offset=0)
        assert candidate is not None
        assert candidate.net_delta == Decimal("0.5") * LOT_SIZE


class TestDebitSpreads:
    def test_call_debit_spread_economics(self, view: structures.ChainView):
        candidate = structures.build_structure(
            StrategyType.CALL_DEBIT_SPREAD, view, anchor_offset=0, width_steps=1
        )
        assert candidate is not None
        # Buy 24500 CE at 102, sell 24600 CE at 60 -> 42 net debit.
        debit = Decimal(42 * LOT_SIZE)
        assert candidate.net_premium == debit
        assert candidate.max_loss == debit
        # Width 100 x lot, less the debit paid.
        assert candidate.max_profit == Decimal(100 * LOT_SIZE) - debit
        assert candidate.breakeven == [Decimal(24542)]
        assert candidate.reward_to_risk == pytest.approx(58 / 42, rel=1e-6)

    def test_put_debit_spread_sells_the_lower_strike(self, view: structures.ChainView):
        candidate = structures.build_structure(
            StrategyType.PUT_DEBIT_SPREAD, view, anchor_offset=0, width_steps=1
        )
        assert candidate is not None
        # Buy 24500 PE at 100, sell 24400 PE at 55 -> 45 net debit.
        assert candidate.net_premium == Decimal(45 * LOT_SIZE)
        assert candidate.breakeven == [Decimal(24455)]
        sold = next(leg for leg in candidate.legs if leg.side is OrderSide.SELL)
        assert sold.contract.strike == Decimal(24400)


class TestCreditSpreads:
    def test_put_credit_spread_economics(self, view: structures.ChainView):
        candidate = structures.build_structure(
            StrategyType.PUT_CREDIT_SPREAD, view, anchor_offset=-1, width_steps=1
        )
        assert candidate is not None
        # Sell 24400 PE at 55, buy 24300 PE at 42 -> 13 net credit.
        credit = Decimal(13 * LOT_SIZE)
        assert candidate.net_premium == -credit
        assert candidate.is_credit
        assert candidate.max_profit == credit
        # Risk is the width less the credit received.
        assert candidate.max_loss == Decimal(100 * LOT_SIZE) - credit
        assert candidate.breakeven == [Decimal(24387)]

    def test_credit_structures_are_collateralized_by_their_max_loss(
        self, view: structures.ChainView
    ):
        candidate = structures.build_structure(
            StrategyType.PUT_CREDIT_SPREAD, view, anchor_offset=-1, width_steps=1
        )
        assert candidate is not None
        assert candidate.capital_required == candidate.max_loss

    def test_call_credit_spread_breakeven_is_above_the_short_strike(
        self, view: structures.ChainView
    ):
        candidate = structures.build_structure(
            StrategyType.CALL_CREDIT_SPREAD, view, anchor_offset=1, width_steps=1
        )
        assert candidate is not None
        # Sell 24600 CE at 60, buy 24700 CE at 34 -> 26 credit.
        assert candidate.net_premium == -Decimal(26 * LOT_SIZE)
        assert candidate.breakeven == [Decimal(24626)]


class TestNeutralDefinedRisk:
    def test_iron_condor_economics(self, view: structures.ChainView):
        candidate = structures.build_structure(
            StrategyType.NEUTRAL_DEFINED_RISK, view, anchor_offset=1, width_steps=1
        )
        assert candidate is not None
        # Sell 24600 CE at 60, buy 24700 CE at 34, sell 24400 PE at 55,
        # buy 24300 PE at 42 -> credit of (60-34) + (55-42) = 39.
        credit = Decimal(39 * LOT_SIZE)
        assert candidate.net_premium == -credit
        assert candidate.max_profit == credit
        assert candidate.max_loss == Decimal(100 * LOT_SIZE) - credit
        assert candidate.breakeven == [Decimal(24639), Decimal(24361)]
        assert len(candidate.legs) == 4

    def test_condor_has_two_calls_and_two_puts(self, view: structures.ChainView):
        candidate = structures.build_structure(
            StrategyType.NEUTRAL_DEFINED_RISK, view, anchor_offset=1, width_steps=1
        )
        assert candidate is not None
        calls = [leg for leg in candidate.legs if leg.contract.option_type is OptionType.CE]
        puts = [leg for leg in candidate.legs if leg.contract.option_type is OptionType.PE]
        assert len(calls) == 2
        assert len(puts) == 2
        assert sum(1 for leg in candidate.legs if leg.side is OrderSide.SELL) == 2


class TestUnbuildableStructures:
    def test_no_trade_never_builds_a_structure(self, view: structures.ChainView):
        assert structures.build_structure(StrategyType.NO_TRADE, view) is None

    def test_a_structure_needing_strikes_outside_the_chain_is_none(
        self, view: structures.ChainView
    ):
        assert (
            structures.build_structure(
                StrategyType.CALL_DEBIT_SPREAD, view, anchor_offset=0, width_steps=40
            )
            is None
        )

    def test_zero_lots_or_zero_width_is_rejected(self, view: structures.ChainView):
        assert structures.build_structure(StrategyType.LONG_CALL, view, lots=0) is None
        assert (
            structures.build_structure(StrategyType.CALL_DEBIT_SPREAD, view, width_steps=0)
            is None
        )

    def test_a_credit_structure_that_would_price_as_a_debit_is_rejected(
        self, view: structures.ChainView
    ):
        """Inverting the legs of a credit spread produces a debit, which is
        not the structure that was asked for — better to return nothing than
        something mislabeled."""
        assert (
            structures.build_structure(
                StrategyType.PUT_CREDIT_SPREAD, view, anchor_offset=-2, width_steps=-1
            )
            is None
        )


class TestMultipleLots:
    def test_economics_scale_linearly_with_lots(self, view: structures.ChainView):
        one = structures.build_structure(StrategyType.LONG_CALL, view, lots=1)
        three = structures.build_structure(StrategyType.LONG_CALL, view, lots=3)
        assert one is not None and three is not None
        assert three.max_loss == one.max_loss * 3
        # Breakeven is a price, so it must NOT scale with size.
        assert three.breakeven == one.breakeven
