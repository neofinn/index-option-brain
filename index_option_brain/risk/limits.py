"""Risk limits, as typed configuration.

These are the numbers that decide whether the system survives a bad week, so
they are explicit, injected, and versionable — never constants buried in the
engine. Defaults are deliberately conservative: the intended failure mode of a
misconfiguration is a trade that does not happen.

Percentages are expressed as fractions of net equity (0.01 == 1%).
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict


class RiskLimits(BaseModel):
    model_config = ConfigDict(frozen=True)

    # ---- Per-trade sizing
    max_risk_per_trade: Decimal = Decimal("0.01")
    """Fraction of equity that may be lost on one trade if it goes to max loss."""
    max_loss_ceiling: Decimal | None = None
    """Optional absolute rupee cap per trade, applied on top of the fraction."""
    min_lots: int = 1
    max_lots: int = 10

    # ---- Daily
    max_daily_loss: Decimal = Decimal("0.03")
    """Once the day's P&L is down this fraction of equity, no new entries."""

    # ---- Portfolio
    max_open_positions: int = 4
    max_positions_per_strategy: int = 2
    max_positions_per_underlying: int = 3
    max_portfolio_exposure: Decimal = Decimal("0.06")
    """Total committed max loss across open positions, as a fraction of equity."""
    max_delta_notional_multiple: Decimal = Decimal("3.0")
    """Gross delta notional allowed, as a multiple of equity.

    A multiple, not a percentage: options give exposure far above the
    premium paid, so three lots of a 90-rupee call cost ~17,500 and carry
    over 20 lakh of delta notional. Anyone reading "1,200%" would assume a
    bug. 3x is deliberately loose — it is a backstop against an
    accidentally enormous book, not a position-sizing rule.
    """
    delta_projection_sigmas: float = 1.0
    """How far ahead the delta limit is tested, in one-sigma moves.

    Checked against *projected* exposure because gamma means a book sized
    correctly at entry is oversized once the thesis works: a 0.383-delta
    call becomes 0.533 delta on a 100-point rally, 39% more exposure that
    nobody chose. A limit tested only at entry is tested at the one moment
    it is guaranteed to pass.
    """
    max_concentration_per_underlying: Decimal = Decimal("0.04")

    # ---- Margin
    max_margin_utilization: Decimal = Decimal("0.60")
    """Fraction of available margin the engine may commit."""

    # ---- Market quality
    min_liquidity_score: float = 0.4
    max_relative_spread: float = 0.06
    """Worst per-leg bid-ask spread, as a fraction of mid. This is the
    slippage control: on a structure entered and exited at the touch, the
    spread is the dominant cost and it is paid twice."""

    # ---- Structure
    allow_undefined_risk: bool = False
    """Long options have unbounded profit but bounded loss, so they are
    defined-risk for sizing. A structure whose max loss cannot be computed at
    all is refused unless this is deliberately turned on."""

    # ---- Event risk
    event_blackout_hours: float = 24.0
    """No new entries within this many hours of a blocking scheduled event."""
