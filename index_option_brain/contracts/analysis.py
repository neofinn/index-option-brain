"""Outputs produced by each read-only "brain" (spec §5-9).

Every brain consumes structured state and returns one of these — never a
bare score, never a direct trading decision. None of these types may carry
strike selection, position sizing, or order data (spec §5: "Must NOT choose
options, select strikes, size positions, or execute orders").

Each type carries the exact fields the spec's Output list names, plus the
structured detail downstream engines provably need — the Regime Engine can't
classify BREAKOUT without a breakout state, and the Strike Engine can't rank
"support/resistance" without levels. Spec §3 forbids passing uncontrolled
variables between modules, so that detail belongs here on the contract
rather than being recomputed or smuggled through side channels.

Score conventions, applied uniformly:
  * Directional scores run -1..+1, where positive is bullish.
  * Magnitude/quality scores (liquidity, participation, confidence) run 0..1.

Both `OptionsAnalysis` and `VolatilityAnalysis` carry a field the spec names
`iv_score`, and they measure different things — see each field's comment.
`VolatilityAnalysis.iv_score` is the one that decides debit versus credit;
`OptionsAnalysis.iv_score` describes the shape of the surface, not its level.
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from index_option_brain.contracts.enums import (
    BreakoutState,
    Direction,
    IvRegime,
    MarketRegimeType,
    VwapRelationship,
)


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

    # Structured detail required downstream (regime classification, strike
    # ranking, scenario confirmation/invalidation conditions).
    volatility_score: float = 0.0
    vwap_relationship: VwapRelationship = VwapRelationship.AT
    vwap_distance_atr: float = 0.0
    breakout_state: BreakoutState = BreakoutState.NONE
    support_levels: list[Decimal] = Field(default_factory=list)
    resistance_levels: list[Decimal] = Field(default_factory=list)
    atr: Decimal | None = None
    day_range_position: float | None = None
    opening_range_position: float | None = None
    gap_pct: float | None = None

    @property
    def composite_score(self) -> float:
        """Equal-weighted blend of the three primary scores. Deliberately not
        a decision — the Signal Engine weighs this against every other
        domain."""
        return (self.trend_score + self.structure_score + self.momentum_score) / 3


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

    advances: int = 0
    declines: int = 0
    unchanged: int = 0
    weighted_change_pct: float = 0.0
    weight_coverage: float = 0.0
    contributions: dict[str, float] = Field(default_factory=dict)


class OptionsAnalysis(BaseModel):
    """Spec §7. OI is never a standalone BUY/SELL signal: this type reports
    *structure and positioning* only. `oi_structure_score` is directional,
    but the Signal Engine must corroborate it against index/breadth evidence
    before it can contribute to a trade."""

    model_config = ConfigDict(frozen=True)

    call_pressure: float
    put_pressure: float
    oi_structure_score: float
    iv_score: float
    """Surface *shape*, from put-minus-call skew: negative means downside
    protection is being bid (fear), positive means upside is being chased.
    This is not a richness measure — that is VolatilityAnalysis.iv_score."""
    liquidity_score: float
    gamma_zones: list[Decimal] = Field(default_factory=list)
    call_walls: list[Decimal] = Field(default_factory=list)
    put_walls: list[Decimal] = Field(default_factory=list)
    confidence: float
    evidence: list[str] = Field(default_factory=list)

    atm_strike: Decimal | None = None
    atm_iv: float | None = None
    iv_skew: float | None = None
    pcr_oi: float | None = None
    pcr_volume: float | None = None
    strike_concentration: float = 0.0
    max_pain_strike: Decimal | None = None
    chain_completeness: float = 0.0


class VolatilityAnalysis(BaseModel):
    """Spec §8."""

    model_config = ConfigDict(frozen=True)

    regime: IvRegime
    expected_move: Decimal
    """One standard deviation to expiry: spot x IV x sqrt(T).

    About 68% of outcomes fall inside +/- this. It is **not** the same number
    as the ATM straddle price, and the two are routinely conflated — see
    `expected_absolute_move`.
    """
    expected_absolute_move: Decimal | None = None
    """E|move| — the expectation of the absolute move, which is what an ATM
    straddle is worth.

    Exactly `expected_move * sqrt(2/pi)`, so about **20% smaller** than one
    sigma. Both are legitimate statistics and they answer different questions:
    one sigma is a containment band, this is an average magnitude. Using the
    straddle number as a one-sigma band picks strikes 20% too close.
    """
    straddle_price: Decimal | None = None
    """The observed ATM straddle mid, when the chain supplies both sides."""
    straddle_divergence: float | None = None
    """How far the observed straddle sits from what ATM IV implies, as a
    fraction.

    Theory says `straddle / (spot x IV x sqrt(T))` is sqrt(2/pi) = 0.7979 for
    any spot, IV and tenor. Measured against the live NIFTY chain it came out
    at 0.8037 — 0.73% off. So a material deviation is not a modelling
    question, it is a data-quality signal: a stale IV, a book too wide to
    mark, or a genuine dislocation. Free, because both numbers are already
    being computed.
    """
    iv_score: float
    """Premium *richness*, from implied versus realized volatility: positive
    means options are expensive (favours collecting premium), negative means
    cheap (favours paying it). The Strategy Engine reads this field to choose
    between credit and debit structures."""
    expansion_score: float
    confidence: float

    atm_iv: float | None = None
    iv_percentile: float | None = None
    realized_volatility: float | None = None
    iv_rv_ratio: float | None = None
    days_to_expiry: float | None = None
    evidence: list[str] = Field(default_factory=list)


class RegimeState(BaseModel):
    """Spec §9."""

    model_config = ConfigDict(frozen=True)

    regime: MarketRegimeType
    confidence: float
    evidence: list[str] = Field(default_factory=list)
    invalidations: list[str] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)


class AnalysisBundle(BaseModel):
    """The four quantitative brain outputs for one MarketState, carried
    together so later engines (scenario, signal, strategy, strike) receive
    them as typed state rather than as loose arguments.

    Spec §3 lists `market_regime`, `active_scenarios`, and `active_signals`
    on MarketState — i.e. derived analysis is meant to be folded back into
    state as the pipeline advances. This bundle is that same pattern applied
    to the brain outputs, which keeps every spec-defined engine signature
    intact.
    """

    model_config = ConfigDict(frozen=True)

    index: IndexAnalysis
    constituents: ConstituentAnalysis
    options: OptionsAnalysis
    volatility: VolatilityAnalysis
