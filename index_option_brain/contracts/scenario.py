"""Spec §10. The Scenario Engine must never collapse analysis straight into
BUY/SELL — it generates competing futures and must permit NO_TRADE /
UNCERTAIN.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from index_option_brain.contracts.enums import Direction, ScenarioKind


class Scenario(BaseModel):
    model_config = ConfigDict(frozen=True)

    scenario_id: str
    kind: ScenarioKind
    name: str
    direction: Direction
    score: float
    confidence: float
    supporting_evidence: list[str] = Field(default_factory=list)
    contradictory_evidence: list[str] = Field(default_factory=list)
    confirmation_conditions: list[str] = Field(default_factory=list)
    invalidation_conditions: list[str] = Field(default_factory=list)

    @property
    def is_tradeable(self) -> bool:
        """RANGE and NO_TRADE describe futures in which taking a directional
        position is not the conclusion. They are still returned and scored —
        they simply don't authorize direction on their own."""
        return self.kind not in (ScenarioKind.NO_TRADE,)
