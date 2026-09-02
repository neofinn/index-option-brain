"""Execution Gate behaviour (spec §16).

Structure: one baseline that passes all sixteen checks, then one test per
check that breaks exactly one thing and asserts which check catches it. That
shape matters — a gate where two checks accidentally cover the same condition
looks thorough until the day the condition changes shape and neither fires.
"""

from __future__ import annotations

import inspect
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from index_option_brain.contracts.decision import TradeDecision
from index_option_brain.contracts.enums import (
    MarketSessionState,
    OrderSide,
    StrategyType,
    TradeDecisionType,
)
from index_option_brain.contracts.instruments import IndexSpec
from index_option_brain.contracts.risk import PortfolioState, RiskDecision, RiskReasonCode
from index_option_brain.execution.execution_gate import (
    DeterministicExecutionGate,
    ExecutionCheck,
    ExecutionContext,
    ExecutionGateConfig,
    ExecutionGateResult,
)
from index_option_brain.risk.limits import RiskLimits
from tests.execution.conftest import (
    LOT_SIZE,
    MAX_LOSS,
    SHORT_STRIKE,
    account,
    context,
    contract,
    decision,
    live_chain,
    open_position,
    quote,
    risk_decision,
    structure,
)

gate = DeterministicExecutionGate()


def check(
    trade: TradeDecision | None = None, ctx: ExecutionContext | None = None
) -> ExecutionGateResult:
    return gate.validate(trade or decision(), ctx or context())


class TestTheBaselinePasses:
    def test_a_sound_decision_in_a_healthy_market_is_approved(self, good_decision, good_context):
        result = gate.validate(good_decision, good_context)
        assert result.approved
        assert result.failed_checks == []
        assert result.passed_all

    def test_approval_produces_one_order_per_leg(self, good_decision, good_context):
        result = gate.validate(good_decision, good_context)
        assert len(result.order_requests) == 2

    def test_every_mandatory_check_is_represented(self):
        """Sixteen checks, and the evidence on success says so — if a check is
        ever deleted the count changes and this fails."""
        assert len(ExecutionCheck) == 16


class TestNoOverridePath:
    def test_validate_takes_no_override_parameter(self):
        """Mirrors the same test on the Risk Engine. The absence of an
        override argument is the mechanism, not the documentation: there must
        be nothing a caller or an agent could pass to widen a check."""
        parameters = set(inspect.signature(DeterministicExecutionGate.validate).parameters)
        assert parameters == {"self", "decision", "context"}

    def test_the_context_carries_nothing_that_relaxes_a_check(self):
        """Everything on the context is an observation of the world. A field
        like `skip_liquidity_check` would make the gate advisory."""
        fields = set(ExecutionContext.model_fields)
        assert fields == {
            "timestamp",
            "session_state",
            "index_spec",
            "chain",
            "account",
            "portfolio",
            "kill_switch_engaged",
            "pending_thesis_ids",
        }

    def test_a_blocked_result_can_never_carry_an_order(self):
        blocked = ExecutionGateResult.blocked([ExecutionCheck.KILL_SWITCH], ["off"])
        assert blocked.order_requests == []
        assert not blocked.approved
        assert not blocked.passed_all


class TestKillSwitch:
    def test_an_engaged_kill_switch_blocks_everything(self):
        result = check(ctx=context(kill_switch=True))
        assert ExecutionCheck.KILL_SWITCH in result.failed_checks
        assert not result.approved
        assert result.order_requests == []


class TestMarketSession:
    @pytest.mark.parametrize(
        "state",
        [
            MarketSessionState.PRE_MARKET,
            MarketSessionState.OPENING,
            MarketSessionState.CLOSED,
        ],
    )
    def test_entries_outside_the_active_session_are_blocked(self, state):
        result = check(ctx=context(session_state=state))
        assert ExecutionCheck.MARKET_SESSION in result.failed_checks

    def test_the_closing_session_is_blocked_by_default(self):
        result = check(ctx=context(session_state=MarketSessionState.CLOSING))
        assert ExecutionCheck.MARKET_SESSION in result.failed_checks

    def test_the_closing_session_can_be_permitted_deliberately(self):
        lenient = DeterministicExecutionGate(
            config=ExecutionGateConfig(
                allow_entry_in_closing=True, entry_cutoff_ist=datetime.max.time()
            )
        )
        result = lenient.validate(
            decision(), context(session_state=MarketSessionState.CLOSING)
        )
        assert ExecutionCheck.MARKET_SESSION not in result.failed_checks

    def test_entries_after_the_cutoff_are_blocked(self):
        """15:10 IST. The last half hour is where spreads widen and closing
        marks distort, and a position opened then has no session left to be
        managed in."""
        late = datetime(2026, 9, 2, 9, 40, tzinfo=UTC)
        result = check(ctx=context(timestamp=late))
        assert ExecutionCheck.MARKET_SESSION in result.failed_checks
        assert "cutoff" in " ".join(result.evidence)

    def test_the_cutoff_is_evaluated_in_ist(self):
        """A UTC-hour cutoff would either block the whole Indian session or
        none of it."""
        just_before = datetime(2026, 9, 2, 9, 29, tzinfo=UTC)  # 14:59 IST
        assert ExecutionCheck.MARKET_SESSION not in check(
            ctx=context(timestamp=just_before)
        ).failed_checks


class TestDecisionValidity:
    @pytest.mark.parametrize(
        "kind",
        [TradeDecisionType.WAIT, TradeDecisionType.REJECT],
    )
    def test_only_an_execute_decision_may_reach_a_broker(self, kind):
        result = check(trade=decision(kind=kind))
        assert ExecutionCheck.DECISION_VALID in result.failed_checks

    def test_a_decision_with_no_structure_is_blocked(self):
        result = check(trade=decision(contracts=[]))
        assert ExecutionCheck.DECISION_VALID in result.failed_checks
        assert "no structure" in " ".join(result.evidence)

    def test_two_structures_is_an_assembly_error(self):
        """Risk sized exactly one structure, so a decision carrying two is
        not a basket order — it is a bug upstream."""
        result = check(trade=decision(contracts=[structure(), structure()]))
        assert ExecutionCheck.DECISION_VALID in result.failed_checks

    def test_a_decision_without_a_thesis_id_is_blocked(self):
        """Without it the resulting position could not be traced back to its
        reasoning, which is what the whole feedback loop depends on."""
        result = check(trade=decision(thesis_id=""))
        assert ExecutionCheck.DECISION_VALID in result.failed_checks
        assert "thesis_id" in " ".join(result.evidence)

    def test_a_zero_max_loss_is_blocked(self):
        result = check(trade=decision(max_loss=Decimal(0)))
        assert ExecutionCheck.DECISION_VALID in result.failed_checks


class TestRiskApproval:
    def test_an_unapproved_decision_is_blocked(self):
        result = check(trade=decision(risk=risk_decision(approved=False)))
        assert ExecutionCheck.RISK_APPROVED in result.failed_checks

    def test_the_rejection_reason_is_carried_through(self):
        """An operator seeing a blocked order needs risk's reason, not just
        the gate's."""
        result = check(trade=decision(risk=risk_decision(approved=False)))
        assert "INSUFFICIENT_RISK_BUDGET" in " ".join(result.evidence)

    def test_an_approval_of_zero_lots_is_blocked(self):
        zero = RiskDecision(
            approved=True,
            reason_codes=[RiskReasonCode.APPROVED],
            max_loss=Decimal(0),
            quantity=0,
            lots=0,
        )
        result = check(trade=decision(risk=zero))
        assert ExecutionCheck.RISK_APPROVED in result.failed_checks

    def test_lots_and_units_must_agree(self):
        """A RiskDecision carries the size twice, in lots and in units. If the
        two disagree, one of them was written in the wrong unit upstream — and
        a request read in the wrong unit is a position 75x too large, which is
        not visible in the number itself."""
        mismatched = RiskDecision(
            approved=True,
            reason_codes=[RiskReasonCode.APPROVED],
            max_loss=MAX_LOSS,
            quantity=2,  # should be 1 x 75
            lots=1,
        )
        result = check(trade=decision(risk=mismatched))
        assert ExecutionCheck.QUANTITY_VALID in result.failed_checks

    def test_the_gate_sizes_from_lots_not_units(self):
        """Reading `quantity` as lots would send 75 lots instead of one."""
        result = check()
        assert all(order.lots == 1 for order in result.order_requests)
        assert all(order.quantity == LOT_SIZE for order in result.order_requests)


class TestInstrumentValidity:
    def test_a_leg_absent_from_the_live_chain_is_blocked(self):
        """Not pedantry: an instrument the exchange is not quoting cannot be
        priced, cannot be checked for liquidity, and may not be tradeable."""
        result = check(ctx=context(chain=[live_chain()[0]]))
        assert ExecutionCheck.INSTRUMENT_VALID in result.failed_checks
        assert "not present in the live chain" in " ".join(result.evidence)

    def test_a_leg_on_the_wrong_underlying_is_blocked(self):
        wrong = structure(short=contract(SHORT_STRIKE, underlying="BANKNIFTY"))
        result = check(trade=decision(contracts=[wrong]))
        assert ExecutionCheck.INSTRUMENT_VALID in result.failed_checks

    def test_a_suspended_instrument_is_blocked(self):
        suspended = structure(short=contract(SHORT_STRIKE, trading_status="suspended"))
        result = check(trade=decision(contracts=[suspended]))
        assert ExecutionCheck.INSTRUMENT_VALID in result.failed_checks


class TestExpiryValidity:
    def test_an_expired_contract_is_blocked(self):
        stale = structure(short=contract(SHORT_STRIKE, expiry=date(2026, 9, 1)))
        result = check(trade=decision(contracts=[stale]))
        assert ExecutionCheck.EXPIRY_VALID in result.failed_checks

    def test_todays_expiry_is_still_tradeable(self):
        """Expiry-day trading is in scope: a weekly is tradeable right up to
        the close, and blocking it would rule out a whole class of strategy."""
        today = structure(short=contract(SHORT_STRIKE, expiry=date(2026, 9, 2)))
        chain = [
            quote(
                contract(SHORT_STRIKE, expiry=date(2026, 9, 2)),
                bid=Decimal("114.40"),
                ask=Decimal("115.00"),
            ),
            live_chain()[1],
        ]
        result = check(trade=decision(contracts=[today]), ctx=context(chain=chain))
        assert ExecutionCheck.EXPIRY_VALID not in result.failed_checks

    def test_expiry_is_compared_in_ist(self):
        """At 23:00 UTC it is already tomorrow in India, and a contract
        expiring "today" by UTC has in fact expired."""
        late_utc = datetime(2026, 9, 8, 23, 0, tzinfo=UTC)  # 04:30 IST on the 9th
        result = check(ctx=context(timestamp=late_utc))
        assert ExecutionCheck.EXPIRY_VALID in result.failed_checks


class TestStrikeValidity:
    def test_a_strike_off_the_listed_grid_is_blocked(self):
        odd = structure(short=contract(Decimal(23917)))
        result = check(trade=decision(contracts=[odd]))
        assert ExecutionCheck.STRIKE_VALID in result.failed_checks
        assert "strike step" in " ".join(result.evidence)

    def test_a_banknifty_strike_step_is_honoured(self):
        """23,950 is a listed NIFTY strike and not a listed BANKNIFTY one, so
        the grid has to come from the index spec rather than a constant."""
        banknifty_spec = IndexSpec(
            symbol="BANKNIFTY",
            name="Nifty Bank",
            lot_size=30,
            tick_size=Decimal("0.05"),
            strike_step=Decimal(100),
        )
        odd = structure(short=contract(Decimal(23950), lot_size=30))
        result = check(
            trade=decision(contracts=[odd]),
            ctx=context(index_spec=banknifty_spec),
        )
        assert ExecutionCheck.STRIKE_VALID in result.failed_checks


class TestLotSizeValidity:
    def test_a_stale_lot_size_is_blocked(self):
        """Lot sizes are revised by exchange circular and NSE's public
        endpoints do not publish them, so a stale one is a live risk. It would
        otherwise silently mis-size every order built from it."""
        stale = structure(short=contract(SHORT_STRIKE, lot_size=50))
        result = check(trade=decision(contracts=[stale]))
        assert ExecutionCheck.LOT_SIZE_VALID in result.failed_checks
        assert "lots of 75" in " ".join(result.evidence)


class TestQuantityValidity:
    def test_more_lots_than_the_per_trade_cap_is_blocked(self):
        result = check(trade=decision(risk=risk_decision(lots=99)))
        assert ExecutionCheck.QUANTITY_VALID in result.failed_checks

    def test_a_zero_ratio_leg_is_blocked(self):
        result = check(trade=decision(contracts=[structure(long_lots=0)]))
        assert ExecutionCheck.QUANTITY_VALID in result.failed_checks

    def test_a_ratio_spread_multiplies_correctly(self):
        """A 2x1 ratio has to reach the broker as two lots against one."""
        ratio = structure(short_lots=2, long_lots=1)
        result = check(trade=decision(contracts=[ratio], risk=risk_decision(lots=2)))
        assert result.approved, result.evidence
        by_side = {order.side: order for order in result.order_requests}
        assert by_side[OrderSide.SELL].lots == 4
        assert by_side[OrderSide.BUY].lots == 2
        assert by_side[OrderSide.SELL].quantity == 4 * LOT_SIZE


class TestPriceValidity:
    def test_a_market_that_moved_beyond_tolerance_is_blocked(self):
        """The structure being priced is no longer the structure that was
        analysed, so its max loss and breakeven no longer hold. Risk approved
        a trade that no longer exists."""
        moved = live_chain(short_bid=Decimal("150.00"), short_ask=Decimal("151.00"))
        result = check(ctx=context(chain=moved))
        assert ExecutionCheck.PRICE_VALID in result.failed_checks
        assert "no longer hold" in " ".join(result.evidence)

    def test_a_small_move_is_tolerated(self):
        nudged = live_chain(short_bid=Decimal("116.00"), short_ask=Decimal("116.60"))
        result = check(ctx=context(chain=nudged))
        assert ExecutionCheck.PRICE_VALID not in result.failed_checks

    def test_a_leg_with_no_live_price_is_blocked(self):
        dead = live_chain(short_bid=None, short_ask=None)
        dead[0] = quote(contract(SHORT_STRIKE), bid=None, ask=None, ltp=Decimal(0))
        result = check(ctx=context(chain=dead))
        assert ExecutionCheck.PRICE_VALID in result.failed_checks

    def test_a_price_off_the_tick_grid_is_blocked(self):
        """NIFTY options tick in 0.05. A price of 114.43 cannot be sent, and
        finding that out from a broker rejection is finding out too late."""
        odd = structure(short_price=Decimal("114.43"))
        result = check(trade=decision(contracts=[odd]))
        assert ExecutionCheck.PRICE_VALID in result.failed_checks
        assert "tick" in " ".join(result.evidence)

    def test_the_deviation_tolerance_is_configurable(self):
        strict = DeterministicExecutionGate(config=ExecutionGateConfig(max_price_deviation=0.001))
        nudged = live_chain(short_bid=Decimal("116.00"), short_ask=Decimal("116.60"))
        result = strict.validate(decision(), context(chain=nudged))
        assert ExecutionCheck.PRICE_VALID in result.failed_checks


class TestLiquidityValidity:
    def test_a_one_sided_quote_is_blocked(self):
        """A leg with no bid can be entered but not exited, only abandoned."""
        one_sided = live_chain(long_bid=None)
        result = check(ctx=context(chain=one_sided))
        assert ExecutionCheck.LIQUIDITY_VALID in result.failed_checks
        assert "not exited" in " ".join(result.evidence)

    def test_thin_open_interest_is_blocked(self):
        thin = live_chain(long_oi=40)
        result = check(ctx=context(chain=thin))
        assert ExecutionCheck.LIQUIDITY_VALID in result.failed_checks

    def test_no_traded_volume_is_blocked(self):
        untraded = live_chain(long_volume=0)
        result = check(ctx=context(chain=untraded))
        assert ExecutionCheck.LIQUIDITY_VALID in result.failed_checks

    def test_the_floors_are_configurable(self):
        lenient = DeterministicExecutionGate(
            config=ExecutionGateConfig(min_open_interest=0, min_traded_volume=0)
        )
        thin = live_chain(long_oi=1, long_volume=1)
        result = lenient.validate(decision(), context(chain=thin))
        assert ExecutionCheck.LIQUIDITY_VALID not in result.failed_checks


class TestSpreadAcceptability:
    def test_a_wide_spread_is_blocked(self):
        """The spread is the dominant cost on a structure entered and exited
        at the touch, and it is paid twice."""
        wide = live_chain(long_bid=Decimal("54.00"), long_ask=Decimal("66.00"))
        result = check(ctx=context(chain=wide))
        assert ExecutionCheck.SPREAD_ACCEPTABLE in result.failed_checks
        assert "paid twice" in " ".join(result.evidence)

    def test_the_ceiling_comes_from_risk_limits(self):
        """One source of truth. A spread ceiling restated in the gate config
        would let risk and execution disagree about the same limit."""
        strict = DeterministicExecutionGate(limits=RiskLimits(max_relative_spread=0.001))
        result = strict.validate(decision(), context())
        assert ExecutionCheck.SPREAD_ACCEPTABLE in result.failed_checks


class TestMarginAvailability:
    def test_insufficient_margin_is_blocked(self):
        poor = account(equity="2000000", available_margin="5000")
        result = check(ctx=context(acct=poor))
        assert ExecutionCheck.MARGIN_AVAILABLE in result.failed_checks

    def test_the_estimate_carries_headroom(self):
        """Margin is SPAN + exposure, computed by the exchange and only
        estimated here. One lot needs about 12,558; with 1.10x headroom that
        is 13,814, so 13,000 of available margin must be refused."""
        tight = account(available_margin="13000")
        result = check(ctx=context(acct=tight))
        assert ExecutionCheck.MARGIN_AVAILABLE in result.failed_checks

    def test_the_utilization_cap_is_enforced(self):
        """Having the margin is not the same as being allowed to commit it."""
        capped = DeterministicExecutionGate(
            limits=RiskLimits(max_margin_utilization=Decimal("0.01"))
        )
        result = capped.validate(decision(), context(acct=account(available_margin="100000")))
        assert ExecutionCheck.MARGIN_AVAILABLE in result.failed_checks
        assert "permitted" in " ".join(result.evidence)


class TestDailyLossLimit:
    def test_a_day_at_the_loss_limit_blocks_new_entries(self):
        """3% of 2,000,000 is 60,000. This is read from the live portfolio, so
        a loss taken by an unrelated position blocks this one."""
        acct = account()
        spent = PortfolioState(account=acct, daily_realized_pnl=Decimal(-60_000))
        result = check(ctx=context(acct=acct, portfolio=spent))
        assert ExecutionCheck.DAILY_LOSS_LIMIT in result.failed_checks

    def test_unrealized_losses_count_toward_the_limit(self):
        """A loss you have not closed is still a loss, and a system that only
        counted realized P&L would keep adding risk all the way down."""
        acct = account()
        drawdown = PortfolioState(
            account=acct,
            daily_realized_pnl=Decimal(-30_000),
            daily_unrealized_pnl=Decimal(-31_000),
        )
        result = check(ctx=context(acct=acct, portfolio=drawdown))
        assert ExecutionCheck.DAILY_LOSS_LIMIT in result.failed_checks

    def test_a_profitable_day_does_not_block(self):
        acct = account()
        good = PortfolioState(account=acct, daily_realized_pnl=Decimal(40_000))
        result = check(ctx=context(acct=acct, portfolio=good))
        assert ExecutionCheck.DAILY_LOSS_LIMIT not in result.failed_checks

    def test_zero_equity_blocks_rather_than_dividing_by_it(self):
        acct = account(equity="0")
        result = check(ctx=context(acct=acct))
        assert ExecutionCheck.DAILY_LOSS_LIMIT in result.failed_checks


class TestPositionLimits:
    def test_the_open_position_cap_is_enforced(self):
        acct = account()
        full = PortfolioState(
            account=acct,
            open_positions=[open_position(thesis_id=f"t{n}") for n in range(4)],
        )
        result = check(ctx=context(acct=acct, portfolio=full))
        assert ExecutionCheck.POSITION_LIMIT in result.failed_checks

    def test_the_per_strategy_cap_is_enforced(self):
        acct = account()
        same_strategy = PortfolioState(
            account=acct,
            open_positions=[
                open_position(thesis_id=f"t{n}", strategy=StrategyType.PUT_CREDIT_SPREAD)
                for n in range(2)
            ],
        )
        result = check(ctx=context(acct=acct, portfolio=same_strategy))
        assert ExecutionCheck.POSITION_LIMIT in result.failed_checks

    def test_the_per_underlying_cap_is_enforced(self):
        acct = account()
        crowded = PortfolioState(
            account=acct,
            open_positions=[
                open_position(thesis_id=f"t{n}", strategy=StrategyType.LONG_CALL)
                for n in range(3)
            ],
        )
        result = check(ctx=context(acct=acct, portfolio=crowded))
        assert ExecutionCheck.POSITION_LIMIT in result.failed_checks

    def test_the_caps_come_from_risk_limits(self):
        one_only = DeterministicExecutionGate(limits=RiskLimits(max_open_positions=1))
        acct = account()
        portfolio = PortfolioState(account=acct, open_positions=[open_position()])
        result = one_only.validate(decision(), context(acct=acct, portfolio=portfolio))
        assert ExecutionCheck.POSITION_LIMIT in result.failed_checks


class TestDuplicateOrders:
    def test_a_thesis_already_open_is_blocked(self):
        acct = account()
        held = PortfolioState(
            account=acct, open_positions=[open_position(thesis_id="thesis-1")]
        )
        result = check(ctx=context(acct=acct, portfolio=held))
        assert ExecutionCheck.DUPLICATE_ORDER_CHECK in result.failed_checks

    def test_a_thesis_with_an_order_in_flight_is_blocked(self):
        """The window between submission and fill is exactly when a duplicate
        is easiest to send: the position book does not show it yet."""
        result = check(ctx=context(pending=frozenset({"thesis-1"})))
        assert ExecutionCheck.DUPLICATE_ORDER_CHECK in result.failed_checks
        assert "in flight" in " ".join(result.evidence)

    def test_a_different_thesis_is_not_a_duplicate(self):
        acct = account()
        other = PortfolioState(
            account=acct, open_positions=[open_position(thesis_id="thesis-other")]
        )
        result = check(ctx=context(acct=acct, portfolio=other))
        assert ExecutionCheck.DUPLICATE_ORDER_CHECK not in result.failed_checks


class TestOrderConstruction:
    def test_the_protective_long_leg_is_sequenced_first(self):
        """On a credit spread the long leg is the protection. Sending the
        short leg first and failing on the long one leaves a naked short —
        the worst outcome available to this system. Indian brokers also grant
        spread margin only once the hedge is present."""
        result = check()
        assert [order.side for order in result.order_requests] == [
            OrderSide.BUY,
            OrderSide.SELL,
        ]
        assert [order.sequence for order in result.order_requests] == [0, 1]

    def test_the_limit_price_is_the_live_mid_not_the_stale_reference(self):
        """The reference price is what the structure was analysed at, and by
        the time the gate runs it is stale by however long the cycle took."""
        moved = live_chain(short_bid=Decimal("116.00"), short_ask=Decimal("116.60"))
        result = check(ctx=context(chain=moved))
        short = next(o for o in result.order_requests if o.side is OrderSide.SELL)
        assert short.limit_price == Decimal("116.30")

    def test_limit_prices_are_tick_aligned(self):
        odd = live_chain(long_bid=Decimal("59.78"), long_ask=Decimal("60.20"))
        result = check(ctx=context(chain=odd))
        for order in result.order_requests:
            assert order.limit_price is not None
            assert order.limit_price % Decimal("0.05") == 0

    def test_every_order_carries_the_thesis_and_decision_ids(self):
        """The thread from reasoning to fill: without both ids on the order,
        a later position cannot be reconciled to the decision that made it."""
        result = check()
        for order in result.order_requests:
            assert order.thesis_id == "thesis-1"
            assert order.decision_id == "decision-1"

    def test_quantity_is_in_units_and_lots_in_lots(self):
        result = check(trade=decision(risk=risk_decision(lots=3)))
        for order in result.order_requests:
            assert order.lots == 3
            assert order.quantity == 3 * LOT_SIZE

    def test_a_single_leg_structure_produces_one_order(self):
        single = structure().model_copy(update={"legs": [structure().legs[1]]})
        chain = [live_chain()[1]]
        result = check(trade=decision(contracts=[single]), ctx=context(chain=chain))
        assert result.approved, result.evidence
        assert len(result.order_requests) == 1
        assert result.order_requests[0].side is OrderSide.BUY


class TestEveryFailureIsReported:
    def test_several_broken_things_all_surface(self):
        """An operator looking at a blocked order needs the whole picture:
        "spread too wide" and "also outside market hours" lead to different
        actions than either alone."""
        result = check(
            trade=decision(kind=TradeDecisionType.REJECT, risk=risk_decision(approved=False)),
            ctx=context(
                session_state=MarketSessionState.CLOSED,
                kill_switch=True,
                chain=live_chain(long_bid=Decimal("40.00"), long_ask=Decimal("80.00")),
            ),
        )
        assert {
            ExecutionCheck.KILL_SWITCH,
            ExecutionCheck.MARKET_SESSION,
            ExecutionCheck.DECISION_VALID,
            ExecutionCheck.RISK_APPROVED,
            ExecutionCheck.SPREAD_ACCEPTABLE,
        } <= set(result.failed_checks)

    def test_market_and_account_failures_surface_together(self):
        """An authorized decision meeting a market that has gone bad in
        several ways at once. Size-dependent checks like margin only run when
        risk authorized a size, so this case is separate from the one above."""
        acct = account(available_margin="8000")
        result = check(
            ctx=context(
                acct=acct,
                chain=live_chain(
                    long_bid=Decimal("40.00"), long_ask=Decimal("80.00"), long_oi=3
                ),
            )
        )
        assert {
            ExecutionCheck.SPREAD_ACCEPTABLE,
            ExecutionCheck.LIQUIDITY_VALID,
            ExecutionCheck.MARGIN_AVAILABLE,
        } <= set(result.failed_checks)
        # PRICE_VALID passes on purpose: a 40/80 book has a mid of 60, exactly
        # where the leg was analysed. A wide market is not a moved market, and
        # conflating the two would make the spread check redundant.
        assert ExecutionCheck.PRICE_VALID not in result.failed_checks

    def test_each_failed_check_appears_once(self):
        result = check(ctx=context(kill_switch=True, session_state=MarketSessionState.CLOSED))
        assert len(result.failed_checks) == len(set(result.failed_checks))

    def test_every_failure_carries_a_reason(self):
        result = check(ctx=context(kill_switch=True))
        assert len(result.evidence) >= len(result.failed_checks)
        assert all(reason.strip() for reason in result.evidence)


class TestFailClosed:
    def test_an_exception_becomes_a_rejection(self):
        """A gate that raises leaves the caller with no answer, and the safe
        answer to "may I send this order" is always no."""

        class Exploding(ExecutionContext):
            def quote_for(self, instrument_key: str):
                raise RuntimeError("chain lookup exploded")

        ctx = Exploding(**context().model_dump())
        result = gate.validate(decision(), ctx)
        assert not result.approved
        assert result.order_requests == []
        assert "exploded" in " ".join(result.evidence)

    def test_the_failure_is_diagnosable(self):
        class Exploding(ExecutionContext):
            def quote_for(self, instrument_key: str):
                raise ValueError("bad key")

        ctx = Exploding(**context().model_dump())
        result = gate.validate(decision(), ctx)
        assert "ValueError" in " ".join(result.evidence)


class TestDeterminism:
    def test_the_same_inputs_give_the_same_answer(self):
        """No clock reads, no randomness, no network — which is what lets the
        same gate run in live, paper, backtest and replay (spec §22)."""
        first = check()
        second = check()
        assert first.approved == second.approved
        assert [o.limit_price for o in first.order_requests] == [
            o.limit_price for o in second.order_requests
        ]

    def test_it_reads_the_context_timestamp_not_the_wall_clock(self):
        """Replaying a 2019 decision must be judged against 2019's session,
        not today's."""
        historic = datetime(2019, 4, 3, 6, 30, tzinfo=UTC)
        result = check(ctx=context(timestamp=historic))
        assert ExecutionCheck.EXPIRY_VALID not in result.failed_checks
        assert ExecutionCheck.MARKET_SESSION not in result.failed_checks
