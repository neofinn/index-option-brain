"""Spec §17 and the §30 order state machine.

Only the Order Manager talks to a broker. No brain module may call a broker
order API: they produce a `TradeDecision`, which reaches an `OrderRequest`
only through the Execution Gate.

What this layer is actually for
------------------------------
Not "send the order" — a broker client does that. It exists for the three
things that go wrong between an authorization and a position, all of which
are silent failures if nobody is watching for them.

**A half-filled structure.** A spread submitted leg by leg can end up with
one leg on and one leg rejected. If the leg that filled was the short one,
the account is holding a naked short option: unbounded risk, from a decision
that authorized a defined-risk spread. The Execution Gate sequences the
protective leg first precisely to avoid this, and this layer refuses to send
the rest of a structure once a leg has failed, cancels whatever is still
working, and reports the exposure explicitly rather than letting it be
discovered from a margin call.

**A lost cancel race.** A cancel can be beaten by a fill. A system that
treats "cancel sent" as "flat" will believe it holds nothing while holding a
position, so CANCEL_PENDING → FILLED is a legal transition here, not an
anomaly to discard.

**A duplicate submission.** A cycle that runs again before the first
acknowledgement arrives would send the same leg twice, and the two would look
like one position of double the size. Submission is therefore keyed on
`OrderRequest.client_order_id`.

Everything here is deterministic and takes its time from an injected clock,
so the same manager runs in live, paper, backtest and replay (spec §22).
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from index_option_brain.contracts.enums import OrderLifecycleState, OrderSide
from index_option_brain.contracts.order import Order, OrderEvent, OrderRequest
from index_option_brain.execution.broker_adapter import BrokerAdapter

Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


# The §30 order state machine, as data rather than as branches. Two entries
# are worth reading twice:
#
#   * SUBMITTED -> FILLED, with no OPEN in between. An aggressive order can
#     fill on arrival, and a machine that insisted on passing through OPEN
#     would reject the acknowledgement of its own successful order.
#   * CANCEL_PENDING -> FILLED and -> PARTIAL. A cancel loses the race often
#     enough that treating it as impossible is how a system comes to believe
#     it is flat while holding a position.
LEGAL_TRANSITIONS: dict[OrderLifecycleState, frozenset[OrderLifecycleState]] = {
    OrderLifecycleState.CREATED: frozenset(
        {OrderLifecycleState.SUBMITTED, OrderLifecycleState.FAILED}
    ),
    OrderLifecycleState.SUBMITTED: frozenset(
        {
            OrderLifecycleState.OPEN,
            OrderLifecycleState.PARTIAL,
            OrderLifecycleState.FILLED,
            OrderLifecycleState.REJECTED,
            OrderLifecycleState.FAILED,
            OrderLifecycleState.CANCEL_PENDING,
        }
    ),
    OrderLifecycleState.OPEN: frozenset(
        {
            OrderLifecycleState.PARTIAL,
            OrderLifecycleState.FILLED,
            OrderLifecycleState.CANCEL_PENDING,
            OrderLifecycleState.CANCELLED,
            OrderLifecycleState.REJECTED,
            OrderLifecycleState.FAILED,
        }
    ),
    OrderLifecycleState.PARTIAL: frozenset(
        {
            OrderLifecycleState.PARTIAL,
            OrderLifecycleState.FILLED,
            OrderLifecycleState.CANCEL_PENDING,
            OrderLifecycleState.CANCELLED,
            OrderLifecycleState.FAILED,
        }
    ),
    OrderLifecycleState.CANCEL_PENDING: frozenset(
        {
            OrderLifecycleState.CANCELLED,
            OrderLifecycleState.FILLED,
            OrderLifecycleState.PARTIAL,
            OrderLifecycleState.FAILED,
        }
    ),
    # Terminal. An order that has filled, been rejected, failed or been
    # cancelled does not change again; a broker message claiming otherwise is
    # about a different order.
    OrderLifecycleState.FILLED: frozenset(),
    OrderLifecycleState.REJECTED: frozenset(),
    OrderLifecycleState.FAILED: frozenset(),
    OrderLifecycleState.CANCELLED: frozenset(),
}

TERMINAL_STATES = frozenset(
    {
        OrderLifecycleState.FILLED,
        OrderLifecycleState.REJECTED,
        OrderLifecycleState.FAILED,
        OrderLifecycleState.CANCELLED,
    }
)

UNSUCCESSFUL_TERMINAL_STATES = frozenset(
    {
        OrderLifecycleState.REJECTED,
        OrderLifecycleState.FAILED,
        OrderLifecycleState.CANCELLED,
    }
)

WORKING_STATES = frozenset(
    {
        OrderLifecycleState.SUBMITTED,
        OrderLifecycleState.OPEN,
        OrderLifecycleState.PARTIAL,
    }
)


class OrderManagerError(RuntimeError):
    """A refusal by this layer, distinct from a broker rejection."""


class IllegalTransition(OrderManagerError):
    """An attempt to move an order somewhere the state machine forbids.

    Raised rather than ignored. A transition the machine does not recognize
    means local state and broker state disagree, and the difference between
    those two is the account's actual position.
    """

    def __init__(
        self, order_id: str, current: OrderLifecycleState, requested: OrderLifecycleState
    ) -> None:
        allowed = sorted(str(s) for s in LEGAL_TRANSITIONS[current])
        super().__init__(
            f"Order {order_id} cannot move {current} -> {requested}. "
            f"Legal from {current}: {allowed or 'nothing, it is terminal'}"
        )
        self.order_id = order_id
        self.current = current
        self.requested = requested


def can_transition(
    current: OrderLifecycleState, requested: OrderLifecycleState
) -> bool:
    return requested in LEGAL_TRANSITIONS[current]


class OrderStore(ABC):
    """Where orders and their transition history live.

    An interface because the audit trail belongs in Postgres (spec §27), but
    the manager must be testable and runnable without it.
    """

    @abstractmethod
    def get(self, order_id: str) -> Order | None: ...

    @abstractmethod
    def find_by_client_id(self, client_order_id: str) -> Order | None: ...

    @abstractmethod
    def save(self, order: Order, client_order_id: str) -> None: ...

    @abstractmethod
    def record(self, event: OrderEvent) -> None: ...

    @abstractmethod
    def events(self, order_id: str) -> list[OrderEvent]: ...

    @abstractmethod
    def all_orders(self) -> list[Order]: ...


class InMemoryOrderStore(OrderStore):
    def __init__(self) -> None:
        self._orders: dict[str, Order] = {}
        self._by_client: dict[str, str] = {}
        self._events: list[OrderEvent] = []

    def get(self, order_id: str) -> Order | None:
        return self._orders.get(order_id)

    def find_by_client_id(self, client_order_id: str) -> Order | None:
        order_id = self._by_client.get(client_order_id)
        return self._orders.get(order_id) if order_id else None

    def save(self, order: Order, client_order_id: str) -> None:
        self._orders[order.order_id] = order
        self._by_client[client_order_id] = order.order_id

    def record(self, event: OrderEvent) -> None:
        self._events.append(event)

    def events(self, order_id: str) -> list[OrderEvent]:
        return [event for event in self._events if event.order_id == order_id]

    def all_orders(self) -> list[Order]:
        return list(self._orders.values())


@dataclass(frozen=True)
class StructureSubmission:
    """The outcome of submitting one multi-leg structure.

    `unhedged_short` is the field that matters. Everything else here is
    description; that one is a call to action.
    """

    orders: list[Order]
    submitted: int
    """How many legs were sent. Fewer than requested means submission was
    stopped deliberately after a failure."""
    requested: int
    aborted_at: int | None = None
    """The `sequence` of the leg that failed, if any."""
    unhedged_short: bool = False
    """Short exposure exists without the protection the decision intended.

    This is the naked-short condition: the account is carrying unbounded risk
    from a decision that authorized a defined-risk structure. It is reported
    rather than silently repaired, because repairing it means placing an order
    the Execution Gate never saw — see `flatten`.
    """
    cancelled: list[str] = field(default_factory=list)
    """Orders cancelled as a consequence of the abort."""
    evidence: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        """Every requested leg is fully filled."""
        return (
            self.submitted == self.requested
            and len(self.orders) == self.requested
            and all(
                order.state is OrderLifecycleState.FILLED for order in self.orders
            )
        )

    @property
    def needs_attention(self) -> bool:
        return self.unhedged_short or self.aborted_at is not None

    @property
    def working(self) -> list[Order]:
        return [order for order in self.orders if order.state in WORKING_STATES]


class OrderManager(ABC):
    @abstractmethod
    async def submit(self, authorization: OrderRequest) -> Order: ...

    @abstractmethod
    async def cancel(self, order_id: str) -> Order: ...

    @abstractmethod
    async def modify(self, order_id: str, request: OrderRequest) -> Order: ...


class DeterministicOrderManager(OrderManager):
    """The production Order Manager.

    Holds no market view and makes no trading decision: it moves authorized
    orders through the §30 state machine and reports what happened. Every
    judgement about *whether* to trade was made before this point.
    """

    def __init__(
        self,
        broker: BrokerAdapter,
        *,
        store: OrderStore | None = None,
        clock: Clock = _utc_now,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._broker = broker
        self._store = store or InMemoryOrderStore()
        self._clock = clock
        self._new_id = id_factory or (lambda: uuid.uuid4().hex[:12])

    # ------------------------------------------------------------- reading

    def get(self, order_id: str) -> Order | None:
        return self._store.get(order_id)

    def events(self, order_id: str) -> list[OrderEvent]:
        return self._store.events(order_id)

    # ---------------------------------------------------------- one order

    async def submit(self, authorization: OrderRequest) -> Order:
        """Send one leg, or return the order already sent for it.

        The idempotency check comes first and is not optional: a cycle that
        re-runs before an acknowledgement arrives would otherwise double the
        position, and the duplicate is indistinguishable from intent
        afterwards.
        """
        existing = self._store.find_by_client_id(authorization.client_order_id)
        if existing is not None:
            return existing

        now = self._clock()
        order = Order(
            order_id=self._new_id(),
            decision_id=authorization.decision_id,
            thesis_id=authorization.thesis_id,
            contract=authorization.contract,
            side=authorization.side,
            quantity=authorization.quantity,
            limit_price=authorization.limit_price,
            state=OrderLifecycleState.CREATED,
            created_at=now,
            updated_at=now,
        )
        self._store.save(order, authorization.client_order_id)
        self._log(order, None, OrderLifecycleState.CREATED, {"origin": "execution_gate"})

        try:
            placed = await self._broker.place_order(authorization)
        except Exception as exc:  # noqa: BLE001 - any transport failure is a failed order
            # FAILED, not REJECTED. The distinction is real: rejected means the
            # broker considered the order and refused it, failed means we do not
            # know whether it arrived — and the second requires reconciliation
            # before anything else is sent.
            failed = self._apply(
                order,
                OrderLifecycleState.FAILED,
                detail={"error": f"{type(exc).__name__}: {exc}"},
                client_order_id=authorization.client_order_id,
            )
            return failed

        submitted = self._apply(
            order,
            OrderLifecycleState.SUBMITTED,
            detail={"broker_order_id": placed.broker_order_id or ""},
            broker_order_id=placed.broker_order_id,
            client_order_id=authorization.client_order_id,
        )
        # The broker may already know more than "submitted".
        if placed.state is not OrderLifecycleState.CREATED and placed.state is not (
            OrderLifecycleState.SUBMITTED
        ):
            submitted = self._adopt(submitted, placed, authorization.client_order_id)
        return submitted

    async def cancel(self, order_id: str) -> Order:
        order = self._require(order_id)
        if order.state in TERMINAL_STATES:
            # Cancelling something already finished is a no-op, not an error:
            # a cancel and a fill racing is normal, and the caller losing that
            # race should not see an exception.
            return order

        pending = self._apply(order, OrderLifecycleState.CANCEL_PENDING)
        if not order.broker_order_id:
            # Never reached the broker, so there is nothing out there to cancel.
            return self._apply(
                pending,
                OrderLifecycleState.CANCELLED,
                detail={"reason": "no broker order id; order never left the system"},
            )
        try:
            result = await self._broker.cancel_order(
                order.broker_order_id, known=order
            )
        except Exception as exc:  # noqa: BLE001
            return self._apply(
                pending,
                OrderLifecycleState.FAILED,
                detail={"error": f"cancel failed: {type(exc).__name__}: {exc}"},
            )
        return self._adopt(pending, result)

    async def modify(self, order_id: str, request: OrderRequest) -> Order:
        """Not implemented, and deliberately not faked as cancel-and-replace.

        Cancel-and-replace is not a modification: between the two the order
        loses its queue position and can be beaten to a fill, so a caller that
        believes it modified an order would be wrong about both its price and
        its priority. A real implementation belongs in the broker adapter that
        supports the operation natively.
        """
        raise OrderManagerError(
            "Order modification requires broker support and is not implemented. "
            "Cancel and submit a new order explicitly if that is what you want "
            "— it is not equivalent, because the replacement loses queue "
            "position."
        )

    async def reconcile(self, order_id: str) -> Order:
        """Re-read one order from the broker and adopt what it says.

        The broker is the authority on order state; local state is a cache of
        it. Reconciliation exists because that cache goes stale in exactly the
        situations that matter — after a FAILED submission, after a lost
        cancel race, after a restart.
        """
        order = self._require(order_id)
        if not order.broker_order_id:
            return order
        try:
            remote = await self._broker.get_order_status(
                order.broker_order_id, known=order
            )
        except Exception as exc:  # noqa: BLE001
            self._log(
                order,
                order.state,
                order.state,
                {"reconcile_error": f"{type(exc).__name__}: {exc}"},
            )
            return order
        return self._adopt(order, remote)

    # ----------------------------------------------------------- structure

    async def submit_structure(
        self, requests: list[OrderRequest]
    ) -> StructureSubmission:
        """Submit a multi-leg structure in sequence, stopping at the first failure.

        The ordering comes from the Execution Gate, which puts risk-reducing
        legs first. It is re-sorted here anyway: the guarantee is a safety
        property, and a safety property that depends on the caller having
        passed a correctly ordered list is not a guarantee.
        """
        if not requests:
            raise OrderManagerError("A structure needs at least one leg")

        ordered = sorted(requests, key=lambda request: request.sequence)
        placed: list[Order] = []
        evidence: list[str] = []
        aborted_at: int | None = None

        for request in ordered:
            order = await self.submit(request)
            placed.append(order)
            if order.state in UNSUCCESSFUL_TERMINAL_STATES:
                aborted_at = request.sequence
                evidence.append(
                    f"Leg {request.sequence} ({request.side} {request.contract.strike} "
                    f"{request.contract.option_type}) ended {order.state}; "
                    f"{len(ordered) - len(placed)} remaining leg(s) were not sent"
                )
                break

        cancelled: list[str] = []
        if aborted_at is not None:
            # Cancelling the manager's own working orders is unambiguously
            # risk-reducing and needs no further authorization.
            for order in placed:
                if order.state in WORKING_STATES:
                    result = await self.cancel(order.order_id)
                    cancelled.append(order.order_id)
                    placed[placed.index(order)] = result
            placed = [self._require(order.order_id) for order in placed]

        unhedged, exposure_evidence = self._assess_exposure(ordered, placed)
        evidence.extend(exposure_evidence)

        return StructureSubmission(
            orders=placed,
            submitted=len(placed),
            requested=len(ordered),
            aborted_at=aborted_at,
            unhedged_short=unhedged,
            cancelled=cancelled,
            evidence=evidence,
        )

    def _assess_exposure(
        self, requests: list[OrderRequest], orders: list[Order]
    ) -> tuple[bool, list[str]]:
        """Is there short exposure without the protection that was intended?

        Measured against what the *decision* intended, not against what was
        sent. A protective leg that was never submitted leaves the position
        just as naked as one that was rejected, and a check that only looked
        at submitted legs would miss the more common case.
        """
        filled_short = sum(
            order.filled_quantity for order in orders if order.side is OrderSide.SELL
        )
        if filled_short == 0:
            return False, []

        filled_long = sum(
            order.filled_quantity for order in orders if order.side is OrderSide.BUY
        )
        intended_long = sum(
            request.quantity for request in requests if request.side is OrderSide.BUY
        )
        if intended_long == 0:
            # The structure never had a long leg. That is a naked short by
            # design, which the Risk Engine only permits with
            # allow_undefined_risk turned on — not this layer's call to make.
            return False, []
        if filled_long >= intended_long:
            return False, []

        return True, [
            (
                f"UNHEDGED SHORT: {filled_short} unit(s) short are filled but "
                f"only {filled_long} of {intended_long} intended long unit(s) "
                "are in place. The account is carrying undefined risk from a "
                "decision that authorized a defined-risk structure. Close the "
                "short leg or complete the hedge now — flatten() does the first."
            )
        ]

    async def flatten(self, submission: StructureSubmission) -> list[Order]:
        """Close the filled legs of a structure that could not be completed.

        Separate from `submit_structure`, and explicit, because a flattening
        order is an order the Execution Gate never saw. Its checks are all
        about *opening* risk, so they do not apply to reducing it — but
        placing a trade the gate never authorized is not something this layer
        should do on its own initiative. The condition is reported; the remedy
        is invoked.
        """
        closing: list[Order] = []
        for order in submission.orders:
            if order.filled_quantity <= 0:
                continue
            reverse = (
                OrderSide.SELL if order.side is OrderSide.BUY else OrderSide.BUY
            )
            request = OrderRequest(
                decision_id=f"{order.decision_id}-flatten",
                thesis_id=order.thesis_id,
                contract=order.contract,
                side=reverse,
                quantity=order.filled_quantity,
                lots=0,
                # No limit: the point is to be out, not to be out at a price.
                # A limit order here can sit unfilled while the exposure it was
                # meant to remove keeps running.
                limit_price=None,
                sequence=len(closing),
            )
            closing.append(await self.submit(request))
        return closing

    # ------------------------------------------------------------ internals

    def _require(self, order_id: str) -> Order:
        order = self._store.get(order_id)
        if order is None:
            raise OrderManagerError(f"Unknown order {order_id!r}")
        return order

    def _apply(
        self,
        order: Order,
        state: OrderLifecycleState,
        *,
        detail: dict[str, str] | None = None,
        filled_quantity: int | None = None,
        average_fill_price: object | None = None,
        broker_order_id: str | None = None,
        client_order_id: str | None = None,
    ) -> Order:
        if not can_transition(order.state, state):
            raise IllegalTransition(order.order_id, order.state, state)

        updates: dict[str, object] = {"state": state, "updated_at": self._clock()}
        if filled_quantity is not None:
            updates["filled_quantity"] = filled_quantity
        if average_fill_price is not None:
            updates["average_fill_price"] = average_fill_price
        if broker_order_id is not None:
            updates["broker_order_id"] = broker_order_id

        updated = order.model_copy(update=updates)
        self._store.save(updated, client_order_id or self._client_id_for(order))
        self._log(updated, order.state, state, detail or {})
        return updated

    def _adopt(
        self, order: Order, remote: Order, client_order_id: str | None = None
    ) -> Order:
        """Take the broker's view of an order, if the machine allows the move."""
        if remote.state is order.state:
            # Not a transition, but fills can still have advanced.
            if remote.filled_quantity != order.filled_quantity:
                updated = order.model_copy(
                    update={
                        "filled_quantity": remote.filled_quantity,
                        "average_fill_price": remote.average_fill_price,
                        "updated_at": self._clock(),
                    }
                )
                self._store.save(
                    updated, client_order_id or self._client_id_for(order)
                )
                self._log(
                    updated,
                    order.state,
                    order.state,
                    {"fill": str(remote.filled_quantity)},
                )
                return updated
            return order
        return self._apply(
            order,
            remote.state,
            detail={"source": "broker"},
            filled_quantity=remote.filled_quantity,
            average_fill_price=remote.average_fill_price,
            broker_order_id=remote.broker_order_id or order.broker_order_id,
            client_order_id=client_order_id,
        )

    def _client_id_for(self, order: Order) -> str:
        return f"{order.decision_id}:{order.order_id}"

    def _log(
        self,
        order: Order,
        from_state: OrderLifecycleState | None,
        to_state: OrderLifecycleState,
        detail: dict[str, str],
    ) -> None:
        self._store.record(
            OrderEvent(
                order_event_id=self._new_id(),
                order_id=order.order_id,
                timestamp=order.updated_at,
                from_state=from_state,
                to_state=to_state,
                detail=detail,
            )
        )
