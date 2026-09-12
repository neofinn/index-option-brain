"""Spec §11. A Signal is evidence of opportunity — it is explicitly NOT an
order, and must never be produced by primitive single-indicator logic.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from index_option_brain.contracts.enums import Direction


class Signal(BaseModel):
    model_config = ConfigDict(frozen=True)

    signal_id: str
    direction: Direction
    score: float
    confidence: float
    selected_scenario_id: str
    evidence: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    confirmation_required: bool
    invalidation_conditions: list[str] = Field(default_factory=list)
