"""Spec §23. The trading engine must never require an AIProvider to exist —
`DeterministicProvider` below is a legitimate, always-available no-op
implementation that lets the rest of the system depend on
`IntelligenceProvider` without caring whether LLM_ENABLED is true.

An AgentAssessment is investigation/reasoning output ONLY. It may not
override risk, the execution gate, position limits, or maximum loss; it may
not disable the kill switch, bypass order validation, or send broker orders.
Nothing in this module gives it a code path to do any of that — it has no
access to RiskEngine, ExecutionGate, or OrderManager.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


class AgentAssessment(BaseModel):
    model_config = ConfigDict(frozen=True)

    summary: str
    supporting_points: list[str] = Field(default_factory=list)
    contradicting_points: list[str] = Field(default_factory=list)
    recommendation: str | None = None
    confidence: float | None = None


class AnalysisContext(Protocol):
    """Whatever read-only context (MarketState, scenarios, trade history, ...)
    an IntelligenceProvider needs to form an AgentAssessment. Deliberately
    loose here — the concrete shape is defined once the agent tool surface
    (spec §24) is implemented."""


class IntelligenceProvider(ABC):
    @abstractmethod
    async def analyze(self, context: AnalysisContext) -> AgentAssessment: ...


class DeterministicProvider(IntelligenceProvider):
    """The default provider when LLM_ENABLED=false. Always available, always
    a no-op — proves the system never requires an AIProvider to exist."""

    async def analyze(self, context: AnalysisContext) -> AgentAssessment:
        return AgentAssessment(summary="LLM disabled: deterministic quantitative brain only.")
