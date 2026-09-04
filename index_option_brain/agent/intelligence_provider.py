"""Spec §23-24. The optional investigation layer, and the wall around it.

The engine must never require an AIProvider to exist, and
`DeterministicProvider` is what proves it: a real, always-available
implementation that lets everything depend on `IntelligenceProvider` without
caring whether `LLM_ENABLED` is true.

What the layer is for
---------------------
The deterministic brains answer *what* the market is doing. They are poor at
three things that are not arithmetic:

* **Saying it in a sentence.** The evidence lists are already structured and
  complete; a person opening the console at 09:20 wants the paragraph.
* **Explaining an anomaly.** When the observed straddle diverges from what IV
  implies, or a detector fails on a state it cannot read, the number says
  something is wrong and cannot say what.
* **Context that is not in the feed.** A policy decision, a global session, a
  rebalance. This is the same gap the four calendar-only triggers have: no
  data source, and fabricating one would be worse than the gap.

Why the output cannot be a decision
-----------------------------------
`AgentAssessment` carries prose and nothing a downstream engine could do
arithmetic on. There is deliberately no score, no size, no direction and no
recommendation field: a recommendation is one step from a decision, and a
confidence number invites being multiplied into one.

The guarantee is not this docstring. `tests/agent/test_agent_cannot_decide.py`
asserts that no module under `brain/`, `risk/` or `execution/` imports this
package at all — so there is no code path from an assessment to a trade, and
adding one fails the suite rather than a review.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


class AgentAssessment(BaseModel):
    """Investigation output: prose, and provenance for it.

    Every field is text. That is the point — an assessment can be read, shown
    and stored, and it cannot be multiplied by anything.
    """

    model_config = ConfigDict(frozen=True)

    summary: str
    supporting_points: list[str] = Field(default_factory=list)
    contradicting_points: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    """What the assessment could not establish.

    Present because an investigation layer that only reports findings is a
    layer that always finds something. Naming the gaps is most of the value
    when the honest answer is that the data does not say.
    """
    sources: list[str] = Field(default_factory=list)
    """Where each claim came from, so a reader can check it rather than
    trust it."""
    provider: str = "deterministic"

    @property
    def is_empty(self) -> bool:
        return not (self.supporting_points or self.contradicting_points)


class AnalysisContext(Protocol):
    """Read-only context an IntelligenceProvider forms an assessment from.

    Loose on purpose: the concrete surface is `AgentTools`, and every method
    there returns a frozen contract object. Nothing reachable from here can
    place an order, size a position, or change a limit.
    """


class IntelligenceProvider(ABC):
    @abstractmethod
    async def analyze(self, context: AnalysisContext) -> AgentAssessment: ...


class DeterministicProvider(IntelligenceProvider):
    """The default when `LLM_ENABLED=false`. Always available.

    Not a stub that raises, and not a placeholder that returns something
    plausible: it returns a truthful, empty assessment. The system running
    normally with this installed is the demonstration that no AIProvider is
    required (spec §23, §35).
    """

    async def analyze(self, context: AnalysisContext) -> AgentAssessment:
        return AgentAssessment(
            summary=(
                "LLM disabled. The deterministic quantitative brain is the "
                "whole decision path, which is the supported configuration "
                "rather than a degraded one."
            ),
            provider="deterministic",
        )
