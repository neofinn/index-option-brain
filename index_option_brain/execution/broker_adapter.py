"""The interface the OrderManager uses to actually reach a broker. Concrete
implementations (e.g. a Zerodha Kite adapter) live in their own module and
must never be conflated with a simulator. A `SimulatedBrokerAdapter` used for
paper trading/backtests belongs beside a real one but must log/report itself
unambiguously as simulated (spec §36).

`cancel_order` and `get_order_status` take the caller's copy of the order as
`known`, because a broker's reply to either does not necessarily identify the
instrument — a cancel acknowledgement is often just an id and a status. The
adapter cannot reconstruct an order it was never told about, and inventing a
contract to fill the gap would put a fabricated instrument into a reconciled
position. The Order Manager holds the real one, so it passes it; the adapter
takes only state and fills from the reply."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from index_option_brain.contracts.order import Order, OrderRequest


class BrokerFill(BaseModel):
    model_config = ConfigDict(frozen=True)

    order_id: str
    fill_price: Decimal
    fill_quantity: int
    timestamp: datetime


class BrokerAdapter(ABC):
    @abstractmethod
    async def place_order(self, request: OrderRequest) -> Order: ...

    @abstractmethod
    async def cancel_order(
        self, broker_order_id: str, *, known: Order | None = None
    ) -> Order: ...

    @abstractmethod
    async def get_order_status(
        self, broker_order_id: str, *, known: Order | None = None
    ) -> Order: ...
