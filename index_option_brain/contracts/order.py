"""Spec §17, §30. Only the Order Manager may hold/transition these — no brain
module may call broker order APIs directly."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from index_option_brain.contracts.enums import OrderLifecycleState, OrderSide
from index_option_brain.contracts.instruments import OptionContractSpec


class Order(BaseModel):
    model_config = ConfigDict(frozen=True)

    order_id: str
    decision_id: str
    thesis_id: str
    contract: OptionContractSpec
    side: OrderSide
    quantity: int
    limit_price: Decimal | None
    state: OrderLifecycleState
    broker_order_id: str | None = None
    filled_quantity: int = 0
    average_fill_price: Decimal | None = None
    created_at: datetime
    updated_at: datetime


class OrderEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    order_event_id: str
    order_id: str
    timestamp: datetime
    from_state: OrderLifecycleState | None
    to_state: OrderLifecycleState
    detail: dict[str, str] = Field(default_factory=dict)


class OrderRequest(BaseModel):
    """What the Execution Gate hands to the Order Manager once every mandatory
    check in spec §16 has passed.

    One request per leg. A multi-leg structure produces several, and their
    `sequence` is not cosmetic — see `ExecutionGate` for why the protective
    long leg must be sent first.
    """

    model_config = ConfigDict(frozen=True)

    decision_id: str
    thesis_id: str
    contract: OptionContractSpec
    side: OrderSide
    quantity: int
    """Quantity in **units**, i.e. lots x lot_size, which is what Indian
    broker APIs take. `lots` carries the same size in lots so the two can
    never be silently confused — a request read in the wrong unit is a
    position 75x too large, and the difference is not visible in the number.
    """
    lots: int
    limit_price: Decimal | None
    sequence: int = 0
    """Submission order within the structure, ascending. Risk-reducing legs
    come first."""

    @property
    def client_order_id(self) -> str:
        """A stable id derived from the decision and the leg's position in it.

        This is what makes submission idempotent: re-running a cycle before
        the first acknowledgement arrives produces the same id, so the Order
        Manager can recognize the resubmission instead of sending the leg
        twice and holding double the intended size.
        """
        return f"{self.decision_id}:{self.sequence}"
