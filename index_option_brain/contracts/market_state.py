"""Spec §3 — the central MarketState contract.

Every brain consumes structured state and returns structured state/evidence.
Nothing downstream of the Market-State Engine may pass "uncontrolled
variables" between modules — if a module needs something, it belongs on
this object or one of its typed sub-states.

MarketState is frozen. The pipeline advances it with the `with_*` helpers,
each returning a new instance, so an analysis stage can never mutate the
snapshot a parallel stage is reading.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from index_option_brain.contracts.analysis import AnalysisBundle, RegimeState
from index_option_brain.contracts.enums import MarketSessionState
from index_option_brain.contracts.events import Event
from index_option_brain.contracts.instruments import (
    Bar,
    ConstituentQuote,
    IndexQuote,
    IndexSpec,
    OptionQuote,
)
from index_option_brain.contracts.position import PositionState
from index_option_brain.contracts.scenario import Scenario
from index_option_brain.contracts.signal import Signal


class OpeningRange(BaseModel):
    """The first N minutes of the session (spec §5 "opening structure")."""

    model_config = ConfigDict(frozen=True)

    high: Decimal
    low: Decimal
    completed: bool


class IndexState(BaseModel):
    """`daily_bars` holds *completed* sessions, oldest first — so
    `daily_bars[-1]` is the previous session, the source of PDH/PDL/PDC
    levels. Today's forming candle is never appended there; it lives in
    `quote` and `intraday_bars`.
    """

    model_config = ConfigDict(frozen=True)

    quote: IndexQuote
    spec: IndexSpec | None = None
    intraday_bars: list[Bar] = Field(default_factory=list)
    daily_bars: list[Bar] = Field(default_factory=list)
    opening_range: OpeningRange | None = None


class ConstituentState(BaseModel):
    model_config = ConfigDict(frozen=True)

    quotes: list[ConstituentQuote] = Field(default_factory=list)
    weights: dict[str, float] = Field(default_factory=dict)
    sectors: dict[str, str] = Field(default_factory=dict)


class SectorState(BaseModel):
    model_config = ConfigDict(frozen=True)

    sector_returns: dict[str, float] = Field(default_factory=dict)
    sector_weights: dict[str, float] = Field(default_factory=dict)


class OptionsState(BaseModel):
    model_config = ConfigDict(frozen=True)

    chain: list[OptionQuote] = Field(default_factory=list)
    expiry: date | None = None
    available_expiries: list[date] = Field(default_factory=list)

    forward: Decimal | None = None
    """The expiry's forward, solved from put-call parity on the live book.

    None when no strike had a two-sided book on both legs — an unmeasured
    forward, never `spot` standing in for one.
    """
    forward_basis: Decimal | None = None
    """Forward minus spot, in index points."""
    forward_excess_basis: Decimal | None = None
    """Basis beyond pure interest carry, in index points.

    The part that carries information. Pure carry is mechanical and tells you
    only the rate and the days left; what the market pays *above* it is
    positioning in the futures. On 3 Sep 2026 this went from -21 points on
    the previous close to +22 intraday — a 43-point swing in one session,
    while spot moved 0.4%.
    """
    forward_strikes_used: int = 0
    """Strikes the parity solve averaged over. One is a quote, not a measure."""


class VolatilityState(BaseModel):
    model_config = ConfigDict(frozen=True)

    realized_estimator: str | None = None
    """How `realized_volatility` was measured, e.g. "yang_zhang".

    Carried because "realized volatility is 11%" is not a fact on its own: a
    20-session Yang-Zhang number and a 90-session close-to-close number over
    the same market can differ by a third, and a comparison against implied
    only means something if both sides describe the same horizon.
    """
    realized_window: int | None = None
    """Sessions the realized measurement covers, matched to the option tenor."""
    volatility_risk_premium: float | None = None
    """Implied minus realized, in volatility points, or None if unmeasured.

    Positive: options price more movement than the index has delivered —
    expensive, favouring sellers. Negative: the index has moved more than
    options price — cheap, and this system's side of the trade.
    """

    india_vix: float | None = None
    india_vix_previous_close: float | None = None
    india_vix_year_high: float | None = None
    india_vix_year_low: float | None = None
    """The 52-week range of India VIX, as the exchange publishes it.

    Worth carrying because it gives implied-volatility *context on the first
    tick*. Ranking ATM IV against its own history needs twenty-odd
    observations, which is weeks of uptime on a feed with no history — until
    then the system has no idea whether premium is historically cheap or dear.
    The exchange publishes this range with every snapshot, for free.
    """

    @property
    def india_vix_percentile(self) -> float | None:
        """Where India VIX sits in its own 52-week range, 0 to 1.

        A level measure, not a richness measure: it says implied volatility is
        low against its own history, not that it is cheap against what the
        index is actually doing. The two are different questions and the
        Volatility brain keeps them apart.
        """
        current = self.india_vix
        high = self.india_vix_year_high
        low = self.india_vix_year_low
        if current is None or high is None or low is None or high <= low:
            return None
        return max(0.0, min(1.0, (current - low) / (high - low)))
    realized_volatility: float | None = None
    atm_iv: float | None = None
    atm_iv_history: list[float] = Field(default_factory=list)
    days_to_expiry: float | None = None


class MarketState(BaseModel):
    """The single object every brain/engine consumes and advances — never via
    ad hoc side channels."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    index_state: IndexState
    constituent_state: ConstituentState
    sector_state: SectorState
    options_state: OptionsState
    volatility_state: VolatilityState
    state_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_state: MarketSessionState = MarketSessionState.ACTIVE
    analysis: AnalysisBundle | None = None
    market_regime: RegimeState | None = None
    active_events: list[Event] = Field(default_factory=list)
    active_scenarios: list[Scenario] = Field(default_factory=list)
    active_signals: list[Signal] = Field(default_factory=list)
    position_state: PositionState = Field(default_factory=PositionState)

    @property
    def index_symbol(self) -> str:
        return self.index_state.quote.symbol

    @property
    def spot(self) -> Decimal:
        return self.index_state.quote.ltp

    def with_events(self, events: list[Event]) -> MarketState:
        return self.model_copy(update={"active_events": events})

    def with_analysis(self, analysis: AnalysisBundle) -> MarketState:
        return self.model_copy(update={"analysis": analysis})

    def with_regime(self, regime: RegimeState) -> MarketState:
        return self.model_copy(update={"market_regime": regime})

    def with_scenarios(self, scenarios: list[Scenario]) -> MarketState:
        return self.model_copy(update={"active_scenarios": scenarios})

    def with_signals(self, signals: list[Signal]) -> MarketState:
        return self.model_copy(update={"active_signals": signals})

    def with_position_state(self, position_state: PositionState) -> MarketState:
        return self.model_copy(update={"position_state": position_state})
