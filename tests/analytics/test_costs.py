"""Transaction costs for Indian index options.

The figures asserted here were computed against the live NIFTY chain and
cross-checked by hand, because the whole reason this module exists is that
costs are large enough relative to a small credit to change which trade is
worth taking.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from index_option_brain.analytics.costs import (
    DEFAULT_COST_MODEL,
    IndianOptionCostModel,
)
from index_option_brain.contracts.enums import OptionType, OrderSide, StrategyType
from index_option_brain.contracts.instruments import OptionContractSpec
from index_option_brain.contracts.strike import StrikeCandidate, StrikeLeg

LOT = 65


def contract(strike: int) -> OptionContractSpec:
    from datetime import date

    return OptionContractSpec(
        underlying_symbol="NIFTY",
        expiry=date(2026, 9, 8),
        strike=Decimal(strike),
        option_type=OptionType.PE,
        lot_size=LOT,
        tick_size=Decimal("0.05"),
    )


class TestOneLeg:
    def test_brokerage_is_flat_not_proportional(self):
        """The structural fact that makes far-OTM selling expensive: a leg
        worth a tenth as much pays the same brokerage."""
        big = DEFAULT_COST_MODEL.leg_cost(
            Decimal(100_000), side=OrderSide.BUY, is_opening=True
        )
        small = DEFAULT_COST_MODEL.leg_cost(
            Decimal(1_000), side=OrderSide.BUY, is_opening=True
        )
        # Both carry the same ₹20 plus GST on it; only the ad-valorem part scales.
        assert small > Decimal(20)
        assert big / small < 5  # not the 100x the turnover ratio would imply

    def test_stt_falls_on_the_sell_side_only(self):
        turnover = Decimal(1953)  # 30.05 x 65
        sell = DEFAULT_COST_MODEL.leg_cost(
            turnover, side=OrderSide.SELL, is_opening=True
        )
        buy = DEFAULT_COST_MODEL.leg_cost(turnover, side=OrderSide.BUY, is_opening=True)
        assert sell > buy
        # 0.1% of premium is the difference, less the buy-side stamp duty.
        assert sell - buy == pytest.approx(
            Decimal("1.953") - turnover * Decimal("0.00003"), abs=Decimal("0.02")
        )

    def test_stamp_duty_only_on_an_opening_buy(self):
        turnover = Decimal(10_000)
        opening = DEFAULT_COST_MODEL.leg_cost(
            turnover, side=OrderSide.BUY, is_opening=True
        )
        closing = DEFAULT_COST_MODEL.leg_cost(
            turnover, side=OrderSide.BUY, is_opening=False
        )
        assert opening > closing

    def test_charges_are_on_premium_not_notional(self):
        """Levied on the money that changes hands. Using the notional of the
        underlying would overstate them by two orders of magnitude."""
        premium_turnover = Decimal("30.05") * LOT  # ~1,953
        cost = DEFAULT_COST_MODEL.leg_cost(
            premium_turnover, side=OrderSide.SELL, is_opening=True
        )
        assert cost < Decimal(30)

    def test_a_zero_leg_costs_nothing(self):
        assert DEFAULT_COST_MODEL.leg_cost(
            Decimal(0), side=OrderSide.BUY, is_opening=True
        ) == Decimal(0)


class TestRoundTrip:
    def test_it_matches_the_measured_live_spread(self):
        """The 23600/23400 put spread on the live chain: sell 30.05, buy 13.40,
        65 lot. Hand-computed at about ₹100."""
        legs = [
            (Decimal("30.05") * LOT, OrderSide.SELL),
            (Decimal("13.40") * LOT, OrderSide.BUY),
        ]
        assert DEFAULT_COST_MODEL.round_trip(legs) == pytest.approx(
            Decimal("99.59"), abs=Decimal("0.5")
        )

    def test_both_halves_are_counted(self):
        """A defined-risk spread is closed, not abandoned. Charging only the
        entry would understate every trade by roughly half."""
        legs = [(Decimal(2_000), OrderSide.SELL)]
        entry_only = DEFAULT_COST_MODEL.leg_cost(
            Decimal(2_000), side=OrderSide.SELL, is_opening=True
        )
        assert DEFAULT_COST_MODEL.round_trip(legs) > entry_only * Decimal("1.5")

    def test_the_closing_leg_reverses_the_side(self):
        """Which matters, because STT falls on the sell: a spread pays it on
        the short leg going in and on the long leg coming out."""
        short_only = DEFAULT_COST_MODEL.round_trip([(Decimal(2_000), OrderSide.SELL)])
        long_only = DEFAULT_COST_MODEL.round_trip([(Decimal(2_000), OrderSide.BUY)])
        # Symmetric on a round trip: each pays exactly one sell-side STT.
        assert short_only == pytest.approx(long_only, abs=Decimal("0.10"))

    def test_the_schedule_is_configurable(self):
        """Rates change by budget and circular, so none of them is a
        literal."""
        free = IndianOptionCostModel(
            brokerage_per_order=Decimal(0),
            stt_sell_rate=Decimal(0),
            exchange_txn_rate=Decimal(0),
            sebi_turnover_rate=Decimal(0),
            stamp_duty_rate=Decimal(0),
            gst_rate=Decimal(0),
        )
        assert free.round_trip([(Decimal(10_000), OrderSide.SELL)]) == Decimal(0)


class TestCostShareGrowsAsCreditShrinks:
    """The finding that motivated the module.

    Measured on the live chain at 6 DTE: costs are 3.7% of max profit on the
    near spread and 14.9% four strikes further out — because brokerage is flat
    while the credit shrinks.
    """

    def spread(self, credit_per_unit: str, width: int = 200) -> StrikeCandidate:
        short_price = Decimal(credit_per_unit) + Decimal("13.40")
        legs = [
            StrikeLeg(
                contract=contract(23600),
                side=OrderSide.SELL,
                lots=1,
                reference_price=short_price,
            ),
            StrikeLeg(
                contract=contract(23600 - width),
                side=OrderSide.BUY,
                lots=1,
                reference_price=Decimal("13.40"),
            ),
        ]
        credit = Decimal(credit_per_unit) * LOT
        cost = DEFAULT_COST_MODEL.round_trip(
            [(leg.reference_price * LOT, leg.side) for leg in legs]
        )
        return StrikeCandidate(
            strategy=StrategyType.PUT_CREDIT_SPREAD,
            legs=legs,
            score=0.5,
            net_premium=-credit,
            net_delta=Decimal(5),
            liquidity_score=0.8,
            worst_relative_spread=0.01,
            capital_required=Decimal(width) * LOT - credit,
            max_loss=Decimal(width) * LOT - credit,
            max_profit=credit,
            round_trip_cost=cost,
        )

    def test_a_thin_credit_gives_up_far_more_to_costs(self):
        fat = self.spread("44.55")   # ~2,896 credit
        thin = self.spread("10.09")  # ~656 credit
        assert fat.cost_share_of_profit is not None
        assert thin.cost_share_of_profit is not None
        assert thin.cost_share_of_profit > fat.cost_share_of_profit * 3

    def test_net_reward_is_always_below_gross(self):
        candidate = self.spread("16.65")
        assert candidate.net_reward_to_risk is not None
        assert candidate.reward_to_risk is not None
        assert candidate.net_reward_to_risk < candidate.reward_to_risk

    def test_max_loss_includes_costs_because_they_are_paid_on_a_loser_too(self):
        candidate = self.spread("16.65")
        assert candidate.net_max_loss == candidate.max_loss + candidate.round_trip_cost

    def test_costs_can_push_a_structure_below_the_acceptance_floor(self):
        """The case where this changes a decision rather than a display. A
        structure at gross 0.42 against a 0.40 floor is acceptable; net of a
        cost share like the live chain's, it is not."""
        from index_option_brain.brain.config import StrategyEngineConfig

        floor = StrategyEngineConfig().min_reward_to_risk
        legs = [
            StrikeLeg(
                contract=contract(23600),
                side=OrderSide.SELL,
                lots=1,
                reference_price=Decimal("60.00"),
            ),
            StrikeLeg(
                contract=contract(23400),
                side=OrderSide.BUY,
                lots=1,
                reference_price=Decimal("15.00"),
            ),
        ]
        # Chosen so gross clears the floor by a whisker.
        max_loss = Decimal(1000)
        max_profit = Decimal(str(round(float(max_loss) * (floor + 0.02), 2)))
        cost = Decimal(40)
        candidate = StrikeCandidate(
            strategy=StrategyType.PUT_CREDIT_SPREAD,
            legs=legs,
            score=0.5,
            net_premium=-max_profit,
            net_delta=Decimal(5),
            liquidity_score=0.8,
            worst_relative_spread=0.01,
            capital_required=max_loss,
            max_loss=max_loss,
            max_profit=max_profit,
            round_trip_cost=cost,
        )
        assert candidate.reward_to_risk is not None
        assert candidate.net_reward_to_risk is not None
        assert candidate.reward_to_risk > floor
        assert candidate.net_reward_to_risk < floor

    def test_unbounded_profit_stays_unbounded(self):
        """A long option has no max profit, and subtracting a cost from that
        must not invent one."""
        legs = [
            StrikeLeg(
                contract=contract(23900),
                side=OrderSide.BUY,
                lots=1,
                reference_price=Decimal("130.00"),
            )
        ]
        candidate = StrikeCandidate(
            strategy=StrategyType.LONG_PUT,
            legs=legs,
            score=0.5,
            net_premium=Decimal(130) * LOT,
            net_delta=Decimal(-30),
            liquidity_score=0.9,
            worst_relative_spread=0.01,
            capital_required=Decimal(130) * LOT,
            max_loss=Decimal(130) * LOT,
            max_profit=None,
            round_trip_cost=Decimal(50),
        )
        assert candidate.net_max_profit is None
        assert candidate.net_reward_to_risk is None
        assert candidate.cost_share_of_profit is None


class TestStructuresCarryTheirCost:
    def test_a_built_structure_prices_its_own_round_trip(self, view):
        """Computed from the legs as priced, at the size requested — not a
        flat estimate."""
        from index_option_brain.brain.structures import build_structure

        candidate = build_structure(
            StrategyType.PUT_CREDIT_SPREAD, view, anchor_offset=-1, width_steps=1
        )
        assert candidate is not None
        assert candidate.round_trip_cost > 0
        assert candidate.net_reward_to_risk is not None
        assert candidate.net_reward_to_risk < candidate.reward_to_risk

    def test_cost_scales_with_size(self, view):
        """Ad-valorem charges scale with lots; brokerage does not. So cost per
        lot falls as size rises, which is the mirror image of the far-OTM
        problem."""
        from index_option_brain.brain.structures import build_structure

        one = build_structure(
            StrategyType.PUT_CREDIT_SPREAD, view, anchor_offset=-1, width_steps=1, lots=1
        )
        ten = build_structure(
            StrategyType.PUT_CREDIT_SPREAD, view, anchor_offset=-1, width_steps=1, lots=10
        )
        assert one is not None and ten is not None
        assert ten.round_trip_cost > one.round_trip_cost
        assert ten.round_trip_cost < one.round_trip_cost * 10
