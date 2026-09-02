"""Spec §17. Only the Order Manager talks to the broker. No brain module may
call broker order APIs directly — they only ever produce a TradeDecision that
flows through the ExecutionGate first."""

from __future__ import annotations

from abc import ABC, abstractmethod

from index_option_brain.contracts.order import Order, OrderRequest


class OrderManager(ABC):
    @abstractmethod
    async def submit(self, authorization: OrderRequest) -> Order: ...

    @abstractmethod
    async def cancel(self, order_id: str) -> Order: ...

    @abstractmethod
    async def modify(self, order_id: str, request: OrderRequest) -> Order: ...
