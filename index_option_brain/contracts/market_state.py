"""Spec §3 — the central MarketState contract.

Every brain consumes structured state and returns structured state/evidence.
Nothing downstream of the Market-State Engine may pass "uncontrolled
variables" between modules — if a module needs something, it belongs on
this object or one of its typed sub-states.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from index_option_brain.contracts.analysis import RegimeState
from index_option_brain.contracts.events import Event
from index_option_brain.contracts.instruments import ConstituentQuote, IndexQuote, OptionQuote
from index_option_brain.contracts.position import PositionState
from index_option_brain.contracts.scenario import Scenario
from index_option_brain.contracts.signal import Signal


class IndexState(BaseModel):
    model_config = ConfigDict(frozen=True)

    quote: IndexQuote
    intraday_high: float | None = None
    intraday_low: float | None = None


class ConstituentState(BaseModel):
    model_config = ConfigDict(frozen=True)

    quotes: list[ConstituentQuote] = Field(default_factory=list)
    weights: dict[str, float] = Field(default_factory=dict)


class SectorState(BaseModel):
    model_config = ConfigDict(frozen=True)

    sector_returns: dict[str, float] = Field(default_factory=dict)


class OptionsState(BaseModel):
    model_config = ConfigDict(frozen=True)

    chain: list[OptionQuote] = Field(default_factory=list)


class VolatilityState(BaseModel):
    model_config = ConfigDict(frozen=True)

    india_vix: float | None = None
    realized_volatility: float | None = None


class MarketState(BaseModel):
    """The single object every brain/engine consumes and (a subset) mutates
    via its own analysis output — never via ad hoc side channels."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    index_state: IndexState
    constituent_state: ConstituentState
    sector_state: SectorState
    options_state: OptionsState
    volatility_state: VolatilityState
    market_regime: RegimeState | None = None
    active_events: list[Event] = Field(default_factory=list)
    active_scenarios: list[Scenario] = Field(default_factory=list)
    active_signals: list[Signal] = Field(default_factory=list)
    position_state: PositionState = Field(default_factory=PositionState)
