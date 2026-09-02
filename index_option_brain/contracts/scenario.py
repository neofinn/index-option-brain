"""Spec §10. The Scenario Engine must never collapse analysis straight into
BUY/SELL — it generates competing futures and must permit NO_TRADE / UNCERTAIN.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from index_option_brain.contracts.enums import Direction


class Scenario(BaseModel):
    model_config = ConfigDict(frozen=True)

    scenario_id: str
    name: str
    direction: Direction
    score: float
    confidence: float
    supporting_evidence: list[str] = Field(default_factory=list)
    contradictory_evidence: list[str] = Field(default_factory=list)
    confirmation_conditions: list[str] = Field(default_factory=list)
    invalidation_conditions: list[str] = Field(default_factory=list)
