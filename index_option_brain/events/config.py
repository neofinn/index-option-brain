"""Thresholds for event detection (spec §4).

Every number here is the answer to "how big does this have to be before it is
worth waking the pipeline". They are configuration rather than constants
because they are a tuning exercise, and because the right value for NIFTY is
not the right value for BANKNIFTY.

Where a threshold can be expressed in ATR it is, because a fixed number of
index points means something different at 12% volatility than at 30%.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class _Config(BaseModel):
    model_config = ConfigDict(frozen=True)


class TriggerEngineConfig(_Config):
    # ---- Price
    price_move_atr: float = 0.5
    """A move worth noticing, in ATR. Falls back to `price_move_pct` when no
    ATR is available — which is the case until bars exist."""
    price_move_pct: float = 0.25
    level_test_atr: float = 0.15
    """How close to a support or resistance level counts as testing it."""
    exceptional_move_pct: float = 2.5
    """A move of this size is a different kind of day, not a bigger version of
    a normal one."""

    # ---- Volatility
    vix_change_pct: float = 5.0
    """Relative change in India VIX, in percent of its own level."""
    atm_iv_change_pct: float = 7.5
    realized_vol_change_pct: float = 15.0

    # ---- Options
    oi_change_ratio: float = 0.20
    """Change in a strike's open interest as a fraction of its prior OI."""
    min_oi_for_change: int = 1_000
    """Below this, a large *ratio* is noise: a strike going from 10 to 20 lots
    has doubled and means nothing."""
    premium_move_pct: float = 12.0
    gamma_concentration_change: float = 0.15
    spread_deterioration_ratio: float = 1.75
    """Median relative spread widening by this multiple."""
    min_spread_for_deterioration: float = 0.005
    """Below this the ratio is unstable — a spread going from 0.1% to 0.2% has
    doubled and is still excellent."""

    # ---- Constituents
    breadth_change: float = 0.25
    """Absolute change in the breadth score."""
    constituent_move_pct: float = 1.5
    min_constituent_weight: float = 3.0
    """Only heavyweights raise a MAJOR_CONSTITUENT_MOVEMENT. A 2% move in a
    0.4%-weight name cannot shift the index and should not wake anything."""
    sector_return_change: float = 0.8
    contribution_change: float = 0.15

    # ---- Volume
    volume_anomaly_ratio: float = 2.5
    min_bars_for_volume_baseline: int = 5

    # ---- Time
    heartbeat_seconds: float = 300.0
    expiry_phase_days: float = 1.0


class SignificanceFilterConfig(_Config):
    min_score: float = 0.35
    """Below this an event is recorded but does not wake the pipeline."""
    cooldown_seconds: dict[str, float] = {}
    """Per-trigger-type overrides, keyed by TriggerType value."""
    default_cooldown_seconds: float = 60.0
    """The same trigger firing on every tick would wake the pipeline
    continuously and the analysis would never finish being useful."""
    always_significant: frozenset[str] = frozenset(
        {
            "MARKET_OPEN",
            "PRE_CLOSE",
            "END_OF_DAY",
            "EXPIRY_PHASE",
            "EXCEPTIONAL_MARKET_EVENT",
            "RBI_EVENT",
            "BUDGET_EVENT_RISK",
            "MAJOR_SCHEDULED_ECONOMIC_EVENT",
            "INDEX_REBALANCE",
        }
    )
    """Triggers that bypass both the score floor and the cooldown.

    Not a convenience: a session boundary or a policy announcement changes
    what every other reading means, and suppressing one because a similar
    event fired recently would be suppressing the most important wake-up of
    the day.
    """
