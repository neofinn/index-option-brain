"""Outputs produced by each read-only "brain" (spec §5-9).

Every brain consumes structured state and returns one of these — never a
bare score, never a direct trading decision. None of these types may carry
strike selection, position sizing, or order data (spec §5: "Must NOT choose
options, select strikes, size positions, or execute orders").
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from index_option_brain.contracts.enums import Direction, MarketRegimeType


class IndexAnalysis(BaseModel):
    """Spec §5."""

    model_config = ConfigDict(frozen=True)

    direction: Direction
    trend_score: float
    structure_score: float
    momentum_score: float
    confidence: float
    evidence: list[str] = Field(default_factory=list)
    invalidations: list[str] = Field(default_factory=list)


class ConstituentAnalysis(BaseModel):
    """Spec §6."""

    model_config = ConfigDict(frozen=True)

    breadth_score: float
    participation_score: float
    leadership_score: float
    concentration_score: float
    sector_scores: dict[str, float] = Field(default_factory=dict)
    top_contributors: list[str] = Field(default_factory=list)
    top_detractors: list[str] = Field(default_factory=list)
    confidence: float
    evidence: list[str] = Field(default_factory=list)


class OptionsAnalysis(BaseModel):
    """Spec §7. OI must never be treated as a standalone BUY/SELL signal —
    that constraint is enforced by callers (Signal/Scenario engines), not by
    this type; this type only reports structure."""

    model_config = ConfigDict(frozen=True)

    call_pressure: float
    put_pressure: float
    oi_structure_score: float
    iv_score: float
    liquidity_score: float
    gamma_zones: list[Decimal] = Field(default_factory=list)
    call_walls: list[Decimal] = Field(default_factory=list)
    put_walls: list[Decimal] = Field(default_factory=list)
    confidence: float
    evidence: list[str] = Field(default_factory=list)


class VolatilityAnalysis(BaseModel):
    """Spec §8."""

    model_config = ConfigDict(frozen=True)

    regime: str
    expected_move: Decimal
    iv_score: float
    expansion_score: float
    confidence: float


class RegimeState(BaseModel):
    """Spec §9."""

    model_config = ConfigDict(frozen=True)

    regime: MarketRegimeType
    confidence: float
    evidence: list[str] = Field(default_factory=list)
    invalidations: list[str] = Field(default_factory=list)
