"""Spec §17, §30. Only the Order Manager may hold/transition these — no brain
module may call broker order APIs directly."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from index_option_brain.contracts.enums import OrderLifecycleState
from index_option_brain.contracts.instruments import OptionContractSpec


class Order(BaseModel):
    model_config = ConfigDict(frozen=True)

    order_id: str
    decision_id: str
    thesis_id: str
    contract: OptionContractSpec
    side: str  # "BUY" | "SELL"
    quantity: int
    limit_price: Decimal | None
    state: OrderLifecycleState
    broker_order_id: str | None = None
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
    check in spec §16 has passed."""

    model_config = ConfigDict(frozen=True)

    decision_id: str
    thesis_id: str
    contract: OptionContractSpec
    side: str
    quantity: int
    limit_price: Decimal | None
