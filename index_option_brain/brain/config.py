"""Tunable parameters for every brain, as typed config objects.

These are the knobs a Learning Engine proposal (spec §20) would eventually
change — so they are explicit, versioned data injected into each brain, never
magic numbers buried in the logic and never mutable global state. A brain
constructed without a config gets these defaults; nothing reads them from a
process-wide singleton.

Defaults are deliberately conservative: thresholds are set so that ambiguous
conditions resolve to NEUTRAL/UNCERTAIN/NO_TRADE rather than to a position.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _Config(BaseModel):
    model_config = ConfigDict(frozen=True)


class IndexBrainConfig(_Config):
    ema_fast: int = 20
    ema_slow: int = 50
    slope_period: int = 20
    rsi_period: int = 14
    atr_period: int = 14
    roc_period: int = 10
    swing_lookback: int = 2
    breakout_lookback: int = 20

    ema_separation_scale: float = 0.02
    """EMA gap (as a fraction of the slow EMA) that maps to a ~0.76 score."""
    slope_atr_scale: float = 0.35
    """Per-bar regression slope, in ATR units, that maps to a ~0.76 score."""
    roc_scale: float = 0.02
    vwap_atr_scale: float = 0.75
    breakout_buffer_atr: float = 0.15
    """How far beyond the range a close must be before it counts as a break —
    keeps a one-tick poke from being reported as a breakout."""

    direction_threshold: float = 0.25
    min_daily_bars: int = 30
    min_intraday_bars: int = 10
    support_resistance_levels: int = 3


class ConstituentBrainConfig(_Config):
    leaders: int = 3
    """How many names count as "leadership" for concentration purposes."""
    top_names: int = 5
    breadth_scale: float = 0.6
    contribution_scale: float = 0.5
    """Weighted index change (%) that maps to a ~0.76 contribution score."""
    unchanged_threshold_pct: float = 0.05
    min_coverage: float = 0.5
    """Minimum fraction of index weight that must be observed before the
    analysis is treated as confident."""


class OptionsBrainConfig(_Config):
    basis_full_scale_points: float = 30.0
    """Excess basis, in index points, that scores +/-1.

    Excess basis is what the futures pay above pure interest carry — the
    positioning component, with the mechanical part removed. 30 points on
    NIFTY is roughly 0.12%, a level reached only when the futures are being
    pushed rather than merely carried. Scaled rather than thresholded so a
    quiet basis contributes a small number instead of nothing.
    """
    basis_min_strikes: int = 3
    """Parity strikes required before the basis is reported at all.

    A forward solved off one or two strikes is a quote, not a measurement,
    and a wide book on either leg moves it more than the signal does.
    """
    atm_window: int = 5
    """Strikes either side of ATM used for pressure/PCR/liquidity measures."""
    wall_count: int = 3
    gamma_zone_count: int = 3
    oi_change_scale: float = 0.25
    """Net OI-change imbalance (as a fraction of window OI) mapping to ~0.76."""
    pressure_scale: float = 0.2
    skew_scale: float = 3.0
    """IV points of put-minus-call skew that map to a ~0.76 score."""
    max_relative_spread: float = 0.05
    """Relative spread at or below which liquidity scores 1.0."""
    min_strikes: int = 5


class VolatilityBrainConfig(_Config):
    vrp_full_scale_points: float = 6.0
    """Volatility points of premium that score +/-1.

    Roughly the spread between a quiet NIFTY week and a stressed one, so a
    genuinely rich surface lands near the top of the range without pinning
    every ordinary day there.
    """
    low_percentile: float = 0.20
    normal_percentile: float = 0.60
    elevated_percentile: float = 0.85
    iv_rv_scale: float = 0.35
    """IV/RV ratio distance from 1.0 that maps to a ~0.76 richness score."""
    expansion_scale: float = 0.15
    min_history: int = 20
    min_rankable_history: int = 5
    """Observations required before IV is ranked against its own history at
    all. A percentile computed from one or two prints is arithmetically valid
    and practically meaningless — it would report the first observation of a
    new series as a volatility extreme."""
    trading_days_per_year: int = 252
    calendar_days_per_year: int = 365
    vix_percentile_richness_weight: float = 0.6
    """How far a VIX-percentile fallback may move the richness score.

    Below 1.0 on purpose. Richness is IV against *realized*, and the VIX
    percentile is IV against its own history — a related question, not the
    same one. IV can sit at the 13th percentile and still be dear if the index
    has gone completely quiet. As a stand-in when realized volatility is
    unavailable it is worth having and worth discounting.
    """
    max_straddle_divergence: float = 0.08
    """How far the observed ATM straddle may sit from what ATM IV implies
    before it is reported as a dislocation.

    The two are linked by a constant — sqrt(2/pi) — for any spot, IV and
    tenor, so a gap between them is never a modelling disagreement. Measured
    live it was 0.73%; 8% is loose enough to absorb a wide ATM book and tight
    enough that a stale IV shows up.
    """


class RegimeEngineConfig(_Config):
    trend_threshold: float = 0.35
    range_threshold: float = 0.20
    expiry_days: float = 1.0
    high_volatility_percentile: float = 0.80
    low_volatility_percentile: float = 0.20
    expansion_threshold: float = 0.35
    min_confidence: float = 0.30
    """Below this the winning regime is reported as UNCERTAIN instead."""
    min_index_confidence: float = 0.10
    """Index measurement coverage below which no regime is classified at all.

    Distinct from `min_confidence`, which judges the winning *score*. This one
    judges whether there was anything to score: every structural candidate is
    derived from index bars, and with no bars the index analysis reports every
    score as 0.0 — which the RANGE candidate reads as "perfectly flat" and the
    LOW_VOLATILITY candidate reads as "perfectly calm". Without this gate an
    empty analysis produces a confident label.
    """
    separation_threshold: float = 0.10
    """Two regimes scoring within this of each other is not a classification."""


class ScenarioEngineConfig(_Config):
    min_score: float = 0.15
    max_scenarios: int = 6


class SignalEngineConfig(_Config):
    min_score: float = 0.35
    min_alignment: float = 0.45
    min_primary_vote: float = 0.15
    """The index domain must support the direction by at least this much.

    Agreement alone is not sufficient: a domain that abstains (votes ~0)
    neither agrees nor disagrees, so a lone non-zero domain scores as
    "unanimous". Requiring the primary domain to actually vote is what stops
    options positioning from carrying a trade on its own (spec §7).
    """
    min_participating_domains: int = 2
    """How many domains must express a view before conviction is credited."""
    min_separation: float = 0.12
    """Required gap between the best and second-best scenario. Competing
    futures that score alike mean the evidence does not distinguish them."""
    confirmation_score: float = 0.55
    """Below this a signal is emitted but flagged confirmation_required."""


class StrategyEngineConfig(_Config):
    rich_iv_score: float = 0.25
    """IV richness at or above which credit structures are preferred."""
    cheap_iv_score: float = -0.25
    min_days_to_expiry_for_long: float = 2.0
    """Buying premium into expiry is a theta trap; below this only defined-risk
    spreads are offered."""
    spread_width_expected_move: float = 1.0
    """Spread width as a multiple of the 1-sigma expected move."""
    min_liquidity_score: float = 0.35
    min_reward_to_risk: float = 0.4
    """Reference reward-to-risk that scores a full mark. Applied as a scoring
    component, not a hard gate — credit structures legitimately run below 1:1
    because they win more often than they lose, so rejecting on raw R:R would
    discard the entire premium-selling family. Weak structures are filtered by
    `min_structure_score` instead, which is one mechanism rather than two."""
    min_structure_score: float = 0.45
    """Score a structure must reach to be preferred over standing aside. Below
    this, NO_TRADE outranks it — a weak candidate must never win by being the
    only candidate."""


class StrikeEngineConfig(_Config):
    max_candidates: int = 5
    directional_target_delta: float = 0.45
    credit_target_delta: float = 0.25
    delta_tolerance: float = 0.25
    max_relative_spread: float = 0.08
    wall_penalty: float = 0.25
    """Score penalty for buying directly into a call/put wall."""
    min_open_interest: int = 5_000
    min_long_leg_delta: float = 0.30
    """Hard floor on |delta| for a bought leg in a **debit** structure. A
    rejection, not a score.

    Below 0.30 a bought option is mostly extrinsic value with little
    directional participation: the index has to travel a long way before the
    position responds at all, theta is charged the whole time, and a
    delta-fit score alone would let such a strike through as merely
    lower-ranked rather than excluded — which still lets it win by being the
    best of a bad set.

    Two exemptions, both because the rule is about *paying premium as the
    trade*:

    * A **credit** structure's long leg is insurance, not the expression. It
      is deliberately bought far out of the money and cheap; requiring 0.30
      there would make every defined-risk credit spread unbuildable and
      leave only naked short options, which is the opposite of safer.
    * A **short** leg at 0.20-0.25 delta is the premium-selling trade working
      as intended.

    Set to 0.0 to disable.
    """
    liquidity_weight: float = 0.35
    delta_fit_weight: float = 0.4
    structure_weight: float = 0.25


class PositionBrainConfig(_Config):
    thesis_break_score: float = -0.2
    """Index composite score against the thesis direction that counts as the
    reason for the trade no longer holding."""
    stop_fraction_of_max_loss: float = 0.8
    liquidity_exit_relative_spread: float = 0.15
    min_days_to_expiry: float = 0.5


class BrainConfig(_Config):
    """One object carrying every brain's parameters — what a strategy version
    (spec §20) would pin."""

    index: IndexBrainConfig = Field(default_factory=IndexBrainConfig)
    constituents: ConstituentBrainConfig = Field(default_factory=ConstituentBrainConfig)
    options: OptionsBrainConfig = Field(default_factory=OptionsBrainConfig)
    volatility: VolatilityBrainConfig = Field(default_factory=VolatilityBrainConfig)
    regime: RegimeEngineConfig = Field(default_factory=RegimeEngineConfig)
    scenario: ScenarioEngineConfig = Field(default_factory=ScenarioEngineConfig)
    signal: SignalEngineConfig = Field(default_factory=SignalEngineConfig)
    strategy: StrategyEngineConfig = Field(default_factory=StrategyEngineConfig)
    strike: StrikeEngineConfig = Field(default_factory=StrikeEngineConfig)
    position: PositionBrainConfig = Field(default_factory=PositionBrainConfig)
