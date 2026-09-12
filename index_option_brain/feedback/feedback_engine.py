"""Spec §19. Must never automatically change live strategy parameters from a
single trade's outcome — it only records structured feedback for the
Learning pipeline (spec §20) to later validate."""

from __future__ import annotations

from abc import ABC, abstractmethod

from index_option_brain.contracts.feedback import TradeFeedback
from index_option_brain.contracts.position import Position


class FeedbackEngine(ABC):
    @abstractmethod
    async def record(self, closed_position: Position) -> TradeFeedback: ...
