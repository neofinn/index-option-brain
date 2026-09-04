"""Spec §21 — PostgreSQL as persistent memory: the source of truth for
market/trade/scenario/lesson memory, strategy versions, decisions, orders,
positions, and feedback. Concrete implementations belong in
`index_option_brain.database` behind SQLAlchemy; this module only pins the
contract every caller (including agent tools) depends on."""

from __future__ import annotations

from abc import ABC, abstractmethod

from index_option_brain.contracts.decision import TradeDecision
from index_option_brain.contracts.feedback import Lesson, TradeFeedback
from index_option_brain.contracts.position import Position


class TradeMemoryRepository(ABC):
    @abstractmethod
    async def save_decision(self, decision: TradeDecision) -> None: ...

    @abstractmethod
    async def save_feedback(self, feedback: TradeFeedback) -> None: ...

    @abstractmethod
    async def save_lesson(self, lesson: Lesson) -> None: ...

    @abstractmethod
    async def get_position_history(self, thesis_id: str) -> list[Position]: ...

    @abstractmethod
    async def get_trade_history(self, limit: int = 20) -> list[TradeFeedback]: ...
