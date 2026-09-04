"""Order Manager behaviour (spec §17, §30).

The happy path is the least interesting part. What this layer exists for is
the three silent failures between an authorization and a position: a
half-filled structure that leaves a naked short, a cancel beaten by a fill,
and a duplicate submission that doubles the size. Most of the tests below are
about those.

The broker is a stub whose behaviour each test sets explicitly, so the
sequencing logic is what is under test rather than any broker's quirks.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from index_option_brain.contracts.enums import OrderLifecycleState, OrderSide
from index_option_brain.contracts.order import Order, OrderRequest
from index_option_brain.execution.broker_adapter import BrokerAdapter
from index_option_brain.execution.order_manager import (
    LEGAL_TRANSITIONS,
    TERMINAL_STATES,
    DeterministicOrderManager,
    IllegalTransition,
    InMemoryOrderStore,
    OrderManagerError,
    can_transition,
)
from tests.execution.conftest import LONG_STRIKE, LOT_SIZE, NOW, SHORT_STRIKE, contract

State = OrderLifecycleState


class StubBroker(BrokerAdapter):
    """A broker whose reply to each leg is set by the test.

    Keyed on (side, strike) so a test can say "the short leg is rejected"
    without caring about ids.
    """

    def __init__(self, **behaviour) -> None:
        self.behaviour: dict[tuple[str, str], object] = behaviour.pop("legs", {})
        self.placed: list[OrderRequest] = []
        self.cancelled: list[str] = []
        self.status_calls: list[str] = []
        self.status: dict[str, Order] = {}
        self.cancel_result: Order | None = None
        self.cancel_error: Exception | None = None
        self._counter = 0

    def _key(self, request: OrderRequest) -> tuple[str, str]:
        return (str(request.side), str(request.contract.strike))

    async def place_order(self, request: OrderRequest) -> Order:
        self.placed.append(request)
        outcome = self.behaviour.get(self._key(request), State.FILLED)
        if isinstance(outcome, Exception):
            raise outcome
        self._counter += 1
        filled = request.quantity if outcome is State.FILLED else 0
        if outcome is State.PARTIAL:
            filled = request.quantity // 2
        return Order(
            order_id="broker-side",
            decision_id=request.decision_id,
            thesis_id=request.thesis_id,
            contract=request.contract,
            side=request.side,
            quantity=request.quantity,
            limit_price=request.limit_price,
            state=outcome,
            broker_order_id=f"B{self._counter}",
            filled_quantity=filled,
            average_fill_price=request.limit_price,
            created_at=NOW,
            updated_at=NOW,
        )

    async def cancel_order(
        self, broker_order_id: str, *, known: Order | None = None
    ) -> Order:
        self.cancelled.append(broker_order_id)
        if self.cancel_error is not None:
            raise self.cancel_error
        if self.cancel_result is not None:
            return self.cancel_result
        return Order(
            order_id="broker-side",
            decision_id="d",
            thesis_id="t",
            contract=contract(SHORT_STRIKE),
            side=OrderSide.SELL,
            quantity=LOT_SIZE,
            limit_price=None,
            state=State.CANCELLED,
            broker_order_id=broker_order_id,
            created_at=NOW,
            updated_at=NOW,
        )

    async def get_order_status(
        self, broker_order_id: str, *, known: Order | None = None
    ) -> Order:
        self.status_calls.append(broker_order_id)
        if broker_order_id in self.status:
            return self.status[broker_order_id]
        raise AssertionError(f"no status configured for {broker_order_id}")


def request(
    *,
    side: OrderSide = OrderSide.BUY,
    strike: Decimal = LONG_STRIKE,
    sequence: int = 0,
    lots: int = 1,
    decision_id: str = "decision-1",
    price: str = "60.00",
) -> OrderRequest:
    return OrderRequest(
        decision_id=decision_id,
        thesis_id="thesis-1",
        contract=contract(strike),
        side=side,
        quantity=lots * LOT_SIZE,
        lots=lots,
        limit_price=Decimal(price),
        sequence=sequence,
    )


def spread() -> list[OrderRequest]:
    """The gate's output for a put credit spread: protection first."""
    return [
        request(side=OrderSide.BUY, strike=LONG_STRIKE, sequence=0),
        request(side=OrderSide.SELL, strike=SHORT_STRIKE, sequence=1, price="114.70"),
    ]


def manager(broker: StubBroker | None = None) -> DeterministicOrderManager:
    ticks = iter(NOW + timedelta(seconds=n) for n in range(1000))
    counter = iter(range(1000))
    return DeterministicOrderManager(
        broker or StubBroker(),
        store=InMemoryOrderStore(),
        clock=lambda: next(ticks),
        id_factory=lambda: f"id{next(counter):03d}",
    )


class TestStateMachine:
    def test_every_state_has_a_transition_table(self):
        assert set(LEGAL_TRANSITIONS) == set(State)

    @pytest.mark.parametrize("state", sorted(TERMINAL_STATES))
    def test_terminal_states_go_nowhere(self, state):
        """A filled, rejected, failed or cancelled order does not change
        again; a broker message claiming otherwise is about a different
        order."""
        assert LEGAL_TRANSITIONS[state] == frozenset()

    def test_an_order_can_fill_straight_from_submitted(self):
        """An aggressive order fills on arrival. A machine insisting on OPEN
        first would reject the acknowledgement of its own successful order."""
        assert can_transition(State.SUBMITTED, State.FILLED)

    def test_a_cancel_can_be_beaten_by_a_fill(self):
        """The most important legal transition here. A system that treats
        "cancel sent" as "flat" will believe it holds nothing while holding a
        position."""
        assert can_transition(State.CANCEL_PENDING, State.FILLED)
        assert can_transition(State.CANCEL_PENDING, State.PARTIAL)

    def test_a_partial_can_receive_more_fills(self):
        assert can_transition(State.PARTIAL, State.PARTIAL)

    def test_an_order_cannot_go_backwards(self):
        assert not can_transition(State.FILLED, State.OPEN)
        assert not can_transition(State.OPEN, State.CREATED)
        assert not can_transition(State.CANCELLED, State.SUBMITTED)

    def test_an_illegal_transition_raises_with_what_was_allowed(self):
        """Raised rather than ignored: an unrecognized transition means local
        and broker state disagree, and that difference is the account's actual
        position."""
        error = IllegalTransition("o1", State.FILLED, State.OPEN)
        assert "cannot move" in str(error)
        assert "terminal" in str(error)


class TestSingleOrder:
    async def test_a_submitted_order_reaches_the_broker(self):
        broker = StubBroker()
        order = await manager(broker).submit(request())
        assert len(broker.placed) == 1
        assert order.state is State.FILLED
        assert order.broker_order_id == "B1"

    async def test_the_transition_history_is_recorded(self):
        """The audit trail is what makes reconciliation possible at all."""
        om = manager()
        order = await om.submit(request())
        states = [str(event.to_state) for event in om.events(order.order_id)]
        assert states[0] == "CREATED"
        assert states[-1] == "FILLED"

    async def test_resubmitting_the_same_leg_does_not_send_it_twice(self):
        """A cycle re-running before an acknowledgement arrives would
        otherwise double the position, and afterwards the duplicate is
        indistinguishable from intent."""
        broker = StubBroker()
        om = manager(broker)
        first = await om.submit(request())
        second = await om.submit(request())
        assert first.order_id == second.order_id
        assert len(broker.placed) == 1

    async def test_different_legs_of_one_decision_are_distinct(self):
        broker = StubBroker()
        om = manager(broker)
        await om.submit(request(sequence=0))
        await om.submit(request(sequence=1, side=OrderSide.SELL, strike=SHORT_STRIKE))
        assert len(broker.placed) == 2

    async def test_a_transport_error_becomes_failed_not_rejected(self):
        """Rejected means the broker considered the order and refused it.
        Failed means we do not know whether it arrived — and only the second
        requires reconciliation before anything else is sent."""
        broker = StubBroker(legs={("BUY", str(LONG_STRIKE)): TimeoutError("no route")})
        order = await manager(broker).submit(request())
        assert order.state is State.FAILED
        assert order.state is not State.REJECTED

    async def test_the_failure_reason_is_kept(self):
        broker = StubBroker(legs={("BUY", str(LONG_STRIKE)): TimeoutError("no route")})
        om = manager(broker)
        order = await om.submit(request())
        detail = " ".join(
            value for event in om.events(order.order_id) for value in event.detail.values()
        )
        assert "TimeoutError" in detail
        assert "no route" in detail

    async def test_a_broker_rejection_is_recorded_as_rejected(self):
        broker = StubBroker(legs={("BUY", str(LONG_STRIKE)): State.REJECTED})
        order = await manager(broker).submit(request())
        assert order.state is State.REJECTED
        assert order.filled_quantity == 0


class TestCancel:
    async def test_a_working_order_is_cancelled(self):
        broker = StubBroker(legs={("BUY", str(LONG_STRIKE)): State.OPEN})
        om = manager(broker)
        order = await om.submit(request())
        cancelled = await om.cancel(order.order_id)
        assert cancelled.state is State.CANCELLED
        assert broker.cancelled == ["B1"]

    async def test_cancelling_a_filled_order_is_a_no_op_not_an_error(self):
        """A cancel and a fill racing is normal, and the caller losing that
        race should not see an exception."""
        broker = StubBroker()
        om = manager(broker)
        order = await om.submit(request())
        assert order.state is State.FILLED
        result = await om.cancel(order.order_id)
        assert result.state is State.FILLED
        assert broker.cancelled == []

    async def test_a_cancel_beaten_by_a_fill_is_adopted(self):
        """The broker says it filled. Local state must follow, because the
        account is long whatever the cancel intended."""
        broker = StubBroker(legs={("BUY", str(LONG_STRIKE)): State.OPEN})
        om = manager(broker)
        order = await om.submit(request())
        broker.cancel_result = Order(
            order_id="broker-side",
            decision_id="decision-1",
            thesis_id="thesis-1",
            contract=contract(LONG_STRIKE),
            side=OrderSide.BUY,
            quantity=LOT_SIZE,
            limit_price=Decimal("60.00"),
            state=State.FILLED,
            broker_order_id="B1",
            filled_quantity=LOT_SIZE,
            average_fill_price=Decimal("60.00"),
            created_at=NOW,
            updated_at=NOW,
        )
        result = await om.cancel(order.order_id)
        assert result.state is State.FILLED
        assert result.filled_quantity == LOT_SIZE

    async def test_an_order_that_never_reached_the_broker_cancels_locally(self):
        broker = StubBroker(legs={("BUY", str(LONG_STRIKE)): TimeoutError("down")})
        om = manager(broker)
        order = await om.submit(request())
        # FAILED is terminal, so the cancel is a no-op rather than a fiction.
        assert (await om.cancel(order.order_id)).state is State.FAILED

    async def test_a_failed_cancel_is_recorded_as_failed(self):
        broker = StubBroker(legs={("BUY", str(LONG_STRIKE)): State.OPEN})
        broker.cancel_error = ConnectionError("broker unreachable")
        om = manager(broker)
        order = await om.submit(request())
        result = await om.cancel(order.order_id)
        assert result.state is State.FAILED

    async def test_cancelling_an_unknown_order_raises(self):
        with pytest.raises(OrderManagerError, match="Unknown order"):
            await manager().cancel("nope")


class TestModify:
    async def test_modification_is_refused_rather_than_faked(self):
        """Cancel-and-replace is not a modification: the replacement loses
        queue position and can be beaten to a fill, so a caller believing it
        modified an order would be wrong about both price and priority."""
        om = manager()
        order = await om.submit(request())
        with pytest.raises(OrderManagerError, match="queue position"):
            await om.modify(order.order_id, request())


class TestStructureSubmission:
    async def test_a_clean_spread_fills_both_legs(self):
        result = await manager().submit_structure(spread())
        assert result.complete
        assert result.submitted == 2
        assert not result.needs_attention
        assert not result.unhedged_short

    async def test_legs_are_sent_in_sequence(self):
        """The protective leg first. This is the ordering the gate produced."""
        broker = StubBroker()
        await manager(broker).submit_structure(spread())
        assert [str(r.side) for r in broker.placed] == ["BUY", "SELL"]

    async def test_the_order_is_enforced_not_assumed(self):
        """A safety property that depends on the caller passing a correctly
        ordered list is not a guarantee."""
        broker = StubBroker()
        await manager(broker).submit_structure(list(reversed(spread())))
        assert [str(r.side) for r in broker.placed] == ["BUY", "SELL"]

    async def test_a_failed_first_leg_stops_the_rest(self):
        """The whole point of sequencing. If the protective leg cannot be
        bought, the short leg must never be sold."""
        broker = StubBroker(legs={("BUY", str(LONG_STRIKE)): State.REJECTED})
        result = await manager(broker).submit_structure(spread())
        assert len(broker.placed) == 1
        assert result.aborted_at == 0
        assert result.submitted == 1
        assert not result.complete
        # Nothing was sold, so nothing is naked.
        assert not result.unhedged_short

    async def test_the_abort_is_explained(self):
        broker = StubBroker(legs={("BUY", str(LONG_STRIKE)): State.REJECTED})
        result = await manager(broker).submit_structure(spread())
        joined = " ".join(result.evidence)
        assert "were not sent" in joined

    async def test_an_empty_structure_is_refused(self):
        with pytest.raises(OrderManagerError, match="at least one leg"):
            await manager().submit_structure([])


class TestTheNakedShortCondition:
    """The failure this layer exists for.

    A defined-risk spread that half-fills on the short side is unbounded risk
    from a decision that authorized bounded risk, and it is invisible until a
    margin call.
    """

    async def test_a_filled_short_without_its_hedge_is_flagged(self):
        broker = StubBroker(
            legs={
                ("BUY", str(LONG_STRIKE)): State.OPEN,  # working, unfilled
                ("SELL", str(SHORT_STRIKE)): State.FILLED,
            }
        )
        result = await manager(broker).submit_structure(spread())
        assert result.unhedged_short
        assert result.needs_attention

    async def test_the_warning_says_what_to_do(self):
        broker = StubBroker(
            legs={
                ("BUY", str(LONG_STRIKE)): State.OPEN,
                ("SELL", str(SHORT_STRIKE)): State.FILLED,
            }
        )
        result = await manager(broker).submit_structure(spread())
        joined = " ".join(result.evidence)
        assert "UNHEDGED SHORT" in joined
        assert "undefined risk" in joined

    async def test_a_partly_filled_hedge_still_counts_as_unhedged(self):
        broker = StubBroker(
            legs={
                ("BUY", str(LONG_STRIKE)): State.PARTIAL,
                ("SELL", str(SHORT_STRIKE)): State.FILLED,
            }
        )
        result = await manager(broker).submit_structure(spread())
        assert result.unhedged_short

    async def test_a_fully_hedged_short_is_not_flagged(self):
        result = await manager().submit_structure(spread())
        assert not result.unhedged_short

    async def test_exposure_is_measured_against_intent_not_against_what_was_sent(self):
        """A protective leg never submitted leaves the position just as naked
        as one that was rejected — and that is the more common case."""
        legs = [
            request(side=OrderSide.SELL, strike=SHORT_STRIKE, sequence=0, price="114.70"),
            request(side=OrderSide.BUY, strike=LONG_STRIKE, sequence=1),
        ]
        broker = StubBroker(legs={("BUY", str(LONG_STRIKE)): State.REJECTED})
        # Sorted by sequence, so the SELL genuinely goes first here.
        result = await manager(broker).submit_structure(legs)
        assert result.unhedged_short

    async def test_a_deliberately_naked_structure_is_not_flagged(self):
        """A single short leg is a naked short by design, which the Risk
        Engine only permits with allow_undefined_risk on. Whether that was
        wise is not this layer's call."""
        naked = [request(side=OrderSide.SELL, strike=SHORT_STRIKE, price="114.70")]
        result = await manager().submit_structure(naked)
        assert not result.unhedged_short


class TestAbortCleanup:
    async def test_working_orders_are_cancelled_after_an_abort(self):
        """Cancelling the manager's own working orders is unambiguously
        risk-reducing and needs no further authorization."""
        broker = StubBroker(
            legs={
                ("BUY", str(LONG_STRIKE)): State.OPEN,
                ("SELL", str(SHORT_STRIKE)): State.REJECTED,
            }
        )
        result = await manager(broker).submit_structure(spread())
        assert result.cancelled
        assert broker.cancelled == ["B1"]

    async def test_nothing_is_cancelled_on_a_clean_submission(self):
        broker = StubBroker()
        result = await manager(broker).submit_structure(spread())
        assert result.cancelled == []
        assert broker.cancelled == []


class TestFlatten:
    async def test_it_reverses_every_filled_leg(self):
        broker = StubBroker(
            legs={
                ("BUY", str(LONG_STRIKE)): State.OPEN,
                ("SELL", str(SHORT_STRIKE)): State.FILLED,
            }
        )
        om = manager(broker)
        result = await om.submit_structure(spread())
        assert result.unhedged_short

        closing = await om.flatten(result)
        assert len(closing) == 1
        assert closing[0].side is OrderSide.BUY  # reversing the filled short
        assert closing[0].quantity == LOT_SIZE

    async def test_it_is_separate_from_submission_on_purpose(self):
        """A flattening order is one the Execution Gate never saw. The
        condition is reported; the remedy is invoked deliberately."""
        broker = StubBroker(
            legs={
                ("BUY", str(LONG_STRIKE)): State.OPEN,
                ("SELL", str(SHORT_STRIKE)): State.FILLED,
            }
        )
        om = manager(broker)
        await om.submit_structure(spread())
        # Two legs sent, and nothing placed to close them until asked.
        assert len(broker.placed) == 2

    async def test_closing_orders_carry_no_limit_price(self):
        """The point is to be out, not to be out at a price. A limit here can
        sit unfilled while the exposure it was meant to remove keeps running."""
        broker = StubBroker(
            legs={
                ("BUY", str(LONG_STRIKE)): State.OPEN,
                ("SELL", str(SHORT_STRIKE)): State.FILLED,
            }
        )
        om = manager(broker)
        result = await om.submit_structure(spread())
        await om.flatten(result)
        closing = broker.placed[-1]
        assert closing.limit_price is None

    async def test_unfilled_legs_are_left_alone(self):
        broker = StubBroker(
            legs={
                ("BUY", str(LONG_STRIKE)): State.OPEN,
                ("SELL", str(SHORT_STRIKE)): State.REJECTED,
            }
        )
        om = manager(broker)
        result = await om.submit_structure(spread())
        assert await om.flatten(result) == []

    async def test_the_closing_order_keeps_the_thesis_id(self):
        """The thread from reasoning to fill survives the remediation, so the
        close can be reconciled against the trade it undoes."""
        broker = StubBroker(
            legs={
                ("BUY", str(LONG_STRIKE)): State.OPEN,
                ("SELL", str(SHORT_STRIKE)): State.FILLED,
            }
        )
        om = manager(broker)
        result = await om.submit_structure(spread())
        closing = await om.flatten(result)
        assert closing[0].thesis_id == "thesis-1"


class TestReconciliation:
    async def test_the_broker_is_the_authority_on_state(self):
        """Local state is a cache of the broker's, and it goes stale in
        exactly the situations that matter."""
        broker = StubBroker(legs={("BUY", str(LONG_STRIKE)): State.OPEN})
        om = manager(broker)
        order = await om.submit(request())
        broker.status["B1"] = order.model_copy(
            update={
                "state": State.FILLED,
                "filled_quantity": LOT_SIZE,
                "average_fill_price": Decimal("60.05"),
            }
        )
        reconciled = await om.reconcile(order.order_id)
        assert reconciled.state is State.FILLED
        assert reconciled.average_fill_price == Decimal("60.05")

    async def test_extra_fills_on_an_unchanged_state_are_adopted(self):
        broker = StubBroker(legs={("BUY", str(LONG_STRIKE)): State.PARTIAL})
        om = manager(broker)
        order = await om.submit(request(lots=2))
        assert order.filled_quantity == LOT_SIZE
        broker.status["B1"] = order.model_copy(update={"filled_quantity": 120})
        reconciled = await om.reconcile(order.order_id)
        assert reconciled.state is State.PARTIAL
        assert reconciled.filled_quantity == 120

    async def test_a_reconcile_failure_leaves_state_untouched(self):
        """Better a stale reading than a wrong one: the fallback is to keep
        what is known and record that the check failed."""
        broker = StubBroker(legs={("BUY", str(LONG_STRIKE)): State.OPEN})
        om = manager(broker)
        order = await om.submit(request())
        reconciled = await om.reconcile(order.order_id)  # no status configured
        assert reconciled.state is State.OPEN
        detail = " ".join(
            value for event in om.events(order.order_id) for value in event.detail.values()
        )
        assert "reconcile_error" in detail or "AssertionError" in detail

    async def test_an_order_that_never_reached_a_broker_needs_no_reconciling(self):
        broker = StubBroker(legs={("BUY", str(LONG_STRIKE)): TimeoutError("x")})
        om = manager(broker)
        order = await om.submit(request())
        assert (await om.reconcile(order.order_id)).state is State.FAILED
        assert broker.status_calls == []


class TestDeterminism:
    async def test_nothing_reads_the_wall_clock(self):
        """An injected clock is what lets the same manager run in live, paper,
        backtest and replay."""
        fixed = datetime(2019, 4, 3, 6, 0, tzinfo=UTC)
        om = DeterministicOrderManager(
            StubBroker(), store=InMemoryOrderStore(), clock=lambda: fixed
        )
        order = await om.submit(request())
        assert order.created_at == fixed
        assert order.updated_at == fixed
