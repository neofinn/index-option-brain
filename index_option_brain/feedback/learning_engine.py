"""Spec §20. Pipeline: TRADE -> FEEDBACK -> LESSON -> VALIDATION -> BACKTEST
-> APPROVAL -> NEW VERSION -> PRODUCTION. No direct automatic production
parameter mutation — every step here produces an artifact for a human (or a
separately-gated approval process) to promote, never a live write to
whatever the Strategy Engine is currently using."""

from __future__ import annotations

from abc import ABC, abstractmethod

from index_option_brain.contracts.feedback import Lesson, TradeFeedback


class LearningEngine(ABC):
    @abstractmethod
    async def derive_lessons(self, feedback: list[TradeFeedback]) -> list[Lesson]: ...

    @abstractmethod
    async def propose_strategy_version(self, validated_lessons: list[Lesson]) -> str:
        """Returns a strategy_version identifier awaiting backtest + approval
        — never applies itself to production."""
        ...
