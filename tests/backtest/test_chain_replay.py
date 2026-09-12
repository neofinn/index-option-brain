"""P&L scoring for a replay driven by real chains.

Every test here guards a way the scorer could invent money.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from index_option_brain.backtest.chain_replay import ChainReplayReport, score_trade
from index_option_brain.contracts.enums import OptionType, OrderSide, StrategyType
from index_option_brain.contracts.instruments import OptionContractSpec
from index_option_brain.contracts.strike import StrikeCandidate, StrikeLeg

EXPIRY = date(2026, 9, 15)


def spec(strike: str, opt: OptionType = OptionType.CE) -> OptionContractSpec:
    return OptionContractSpec(
        underlying_symbol="NIFTY",
        expiry=EXPIRY,
        strike=Decimal(strike),
        option_type=opt,
        lot_size=65,
        tick_size=Decimal("0.05"),
    )


def candidate(*legs: StrikeLeg, strategy: StrategyType = StrategyType.LONG_CALL) -> StrikeCandidate:
    return StrikeCandidate(
        strategy=strategy,
        legs=list(legs),
        score=1.0,
        net_premium=Decimal(0),
        net_delta=Decimal(0),
        liquidity_score=1.0,
        worst_relative_spread=0.0,
        capital_required=Decimal(0),
        max_loss=Decimal(0),
        max_profit=None,
    )


def leg(strike: str, side: OrderSide, price: str, opt: OptionType = OptionType.CE) -> StrikeLeg:
    return StrikeLeg(
        contract=spec(strike, opt), side=side, lots=1, reference_price=Decimal(price)
    )


def scored(cand: StrikeCandidate, exits: dict[tuple[date, Decimal, OptionType], Decimal]):
    return score_trade(
        cand,
        entry_day=date(2026, 9, 10),
        exit_day=date(2026, 9, 15),
        exit_prices=exits,
        exit_reason="expiry",
    )


class TestScoring:
    def test_a_long_call_that_doubles_pays_the_move_times_the_lot(self) -> None:
        trade = scored(
            candidate(leg("23500", OrderSide.BUY, "77")),
            {(EXPIRY, Decimal(23500), OptionType.CE): Decimal(154)},
        )
        assert trade is not None
        assert trade.gross == Decimal(77) * 65
        assert trade.costs > 0
        assert trade.net == trade.gross - trade.costs

    def test_a_long_call_expiring_worthless_loses_exactly_the_premium(self) -> None:
        trade = scored(
            candidate(leg("23500", OrderSide.BUY, "77")),
            {(EXPIRY, Decimal(23500), OptionType.CE): Decimal(0)},
        )
        assert trade is not None
        assert trade.gross == Decimal(-77) * 65

    def test_a_short_call_earns_the_decay_and_pays_stt_on_the_way_in(self) -> None:
        trade = scored(
            candidate(leg("23500", OrderSide.SELL, "77")),
            {(EXPIRY, Decimal(23500), OptionType.CE): Decimal(20)},
        )
        assert trade is not None
        assert trade.gross == Decimal(57) * 65

    def test_an_untraded_leg_at_exit_is_not_priced_at_zero(self) -> None:
        """Zero is a specific claim: a total loss long, a perfect win short.

        An OTM strike that simply did not trade on the exit session must not
        be scored as expiring worthless — that is how a backtest manufactures
        its best trades.
        """
        assert scored(candidate(leg("21700", OrderSide.SELL, "3")), {}) is None

    def test_one_missing_leg_voids_the_whole_structure(self) -> None:
        """A spread priced on one leg is not a spread; it is a naked position
        with the hedge silently removed, and it would score as one."""
        trade = scored(
            candidate(
                leg("23500", OrderSide.BUY, "77"),
                leg("23700", OrderSide.SELL, "30"),
            ),
            {(EXPIRY, Decimal(23500), OptionType.CE): Decimal(154)},
        )
        assert trade is None

    def test_costs_are_charged_on_each_side_at_its_own_premium(self) -> None:
        """`round_trip` assumes one premium for both halves. On a long option
        that expires worthless the exit turnover is nil, and charging the
        entry premium twice would overstate the loss."""
        worthless = scored(
            candidate(leg("23500", OrderSide.BUY, "77")),
            {(EXPIRY, Decimal(23500), OptionType.CE): Decimal(0)},
        )
        doubled = scored(
            candidate(leg("23500", OrderSide.BUY, "77")),
            {(EXPIRY, Decimal(23500), OptionType.CE): Decimal(154)},
        )
        assert worthless is not None and doubled is not None
        assert worthless.costs < doubled.costs


class TestReport:
    def test_a_correct_call_can_still_lose_money_to_charges(self) -> None:
        """The failure mode that actually kills Indian options strategies.

        Rs 20 flat brokerage per order plus GST is ~Rs 56 on a one-lot round
        trip regardless of size. A half-rupee move on a 65-unit lot grosses
        Rs 32.50 — the direction was right and the trade still loses. This is
        why `cost_share_of_gross` is reported next to the net: a strategy can
        be predictive and unprofitable at the same time, and only the ratio
        shows which one is wrong.
        """
        report = ChainReplayReport()
        for _ in range(3):
            trade = scored(
                candidate(leg("23500", OrderSide.BUY, "77")),
                {(EXPIRY, Decimal(23500), OptionType.CE): Decimal("77.50")},
            )
            assert trade is not None
            report.trades.append(trade)
        assert report.gross > 0
        assert report.hit_rate == 0.0
        assert report.net < 0
        assert report.cost_share_of_gross is not None
        assert report.cost_share_of_gross > 1.0

    def test_an_empty_report_reports_no_hit_rate_rather_than_zero(self) -> None:
        report = ChainReplayReport()
        assert report.hit_rate is None
        assert report.cost_share_of_gross is None
