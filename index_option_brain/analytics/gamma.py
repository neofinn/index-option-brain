"""Gamma against theta: what a long option needs from the market to hold.

The question this answers is usually posed as an empirical one — "how far
does NIFTY have to move for gamma profits to beat theta decay?" — and it has
a closed-form answer that makes the calculation unnecessary.

Gamma P&L over a move is `0.5 * gamma * move^2`. Theta P&L over a period is
`theta * days`. Setting them equal gives the breakeven move:

    breakeven = sqrt(2 * |theta| * days / gamma)

and for a Black-Scholes option that quantity is **identically the implied
move**, `spot * iv * sqrt(days / 365)`. It falls out of the pricing PDE,
where `theta = -0.5 * spot^2 * iv^2 * gamma` once the rate terms are set
aside. Verified numerically in `test_gamma.py` at 1, 3, 7 and 30 days: the
ratio is 1.0000 every time, not approximately.

So there is no per-position calculation to do, and no strike or expiry that
is "better for gamma". The model has already priced the trade-off, and every
option on the board sits at exactly the same breakeven. What decides the
outcome is one thing:

    **realized volatility above implied, by enough to cover costs.**

Which is the volatility risk premium in `analytics.realized`, seen from the
hedging side rather than the pricing side. A negative premium — the index
moving more than options price — is the condition under which a long gamma
position pays, and it is the same condition under which buying premium pays
at all.

Costs are the part that is not an identity
------------------------------------------
The breakeven above is a frictionless statement. Delta-hedging N times a day
costs N round trips, and on Indian options that is a flat 20 rupees per
order plus levies, plus the spread crossed each time. This module prices
that drag, because it is what turns a mathematically fair trade into a
losing one and it is the term the closed form omits.

Scope
-----
Delta-hedging an index option needs a futures leg, and futures are out of
scope for this system. So gamma scalping is **not implementable here for
NIFTY**, and this module does not pretend otherwise: it exists because the
breakeven is the right way to judge whether a long option is worth holding
at all, hedged or not. On a venue with perpetual futures — Delta Exchange
has them — the hedged version becomes buildable, and the same numbers apply.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal

from index_option_brain.analytics.costs import IndianOptionCostModel
from index_option_brain.contracts.enums import OrderSide

TRADING_DAYS_PER_YEAR = 252
CALENDAR_DAYS_PER_YEAR = 365


def breakeven_move(*, gamma: float, theta: float, days: float = 1.0) -> float | None:
    """Spot move over `days` at which gamma P&L equals theta decay.

    `theta` is per calendar day and negative for a long option; its sign is
    ignored here because the question is about magnitude. Returns None when
    gamma is zero or absent — a position with no convexity has no move that
    rescues it, which is not the same as a breakeven of zero.
    """
    if gamma <= 0 or days <= 0:
        return None
    return math.sqrt(2.0 * abs(theta) * days / gamma)


def implied_move(*, spot: float, iv_percent: float, days: float = 1.0) -> float:
    """The move the option's own volatility prices over `days`.

    Calendar days, matching theta. Using trading days here would make the
    identity below fail by about 25% and look like a modelling error.
    """
    return spot * (iv_percent / 100.0) * math.sqrt(days / CALENDAR_DAYS_PER_YEAR)


@dataclass(frozen=True)
class HedgeDrag:
    """What dynamic hedging costs before it earns anything.

    The term the closed-form breakeven omits, and the one that decides real
    outcomes. Hedging more often tracks delta better and costs linearly more,
    so there is an optimum that has nothing to do with the greeks.
    """

    hedges_per_day: int
    cost_per_hedge: Decimal
    spread_per_hedge: Decimal

    @property
    def daily_cost(self) -> Decimal:
        """Rupees per day for the whole position, not per unit."""
        return (self.cost_per_hedge + self.spread_per_hedge) * self.hedges_per_day

    def move_needed(
        self, *, gamma: float, theta: float, units: int
    ) -> float | None:
        """Move that covers theta *and* the hedging bill.

        `units` is required, and its absence was a real bug here. Theta and
        gamma are quoted per one unit of the underlying, while `daily_cost`
        is rupees for the entire position — adding them directly compares
        premium points to rupees, which silently inflated the breakeven by
        the lot size. On a 65-unit NIFTY lot it turned a 176-point bar into
        1,038 points, which reads as "gamma never pays" rather than as an
        arithmetic error.

        Both sides are put in per-unit terms:

            0.5 * gamma * move^2 * units = |theta| * units + daily_cost

        so the cost divides by units before it can be compared to theta.
        """
        if gamma <= 0 or units <= 0:
            return None
        per_unit_cost = float(self.daily_cost) / units
        total = abs(theta) + per_unit_cost
        return math.sqrt(2.0 * total / gamma)


def hedge_drag(
    *,
    hedges_per_day: int,
    hedge_notional: Decimal,
    spread_fraction: float,
    cost_model: IndianOptionCostModel | None = None,
) -> HedgeDrag:
    """Price one day of dynamic hedging.

    `spread_fraction` is the half-spread crossed on each hedge as a fraction
    of the hedged notional — the cost of trading at the touch rather than
    the mid. It is usually the larger term, and it is the one a brokerage
    comparison never mentions.
    """
    model = cost_model or IndianOptionCostModel()
    per_hedge = model.leg_cost(
        hedge_notional, side=OrderSide.BUY, is_opening=True
    ) + model.leg_cost(hedge_notional, side=OrderSide.SELL, is_opening=False)
    return HedgeDrag(
        hedges_per_day=max(hedges_per_day, 0),
        cost_per_hedge=per_hedge,
        spread_per_hedge=hedge_notional * Decimal(str(spread_fraction)),
    )


@dataclass(frozen=True)
class GammaAssessment:
    """Whether a long option's convexity is worth its decay, given the tape.

    Reports the frictionless breakeven, the implied move it must equal, the
    realized move actually being delivered, and the move once hedging costs
    are included. The verdict is about realized against implied, because the
    first two are the same number by construction.
    """

    spot: float
    gamma: float
    theta: float
    iv_percent: float
    realized_percent: float | None
    units: int = 1
    """Contract units the position holds, needed to compare costs to theta.

    Defaults to 1 so a frictionless assessment needs no size, but any
    assessment carrying a `drag` must set it — see HedgeDrag.move_needed.
    """
    drag: HedgeDrag | None = None

    @property
    def frictionless_breakeven(self) -> float | None:
        return breakeven_move(gamma=self.gamma, theta=self.theta, days=1.0)

    @property
    def implied_daily_move(self) -> float:
        return implied_move(spot=self.spot, iv_percent=self.iv_percent, days=1.0)

    @property
    def realized_daily_move(self) -> float | None:
        """What the index has actually been delivering per day.

        None when realized volatility was not measured — an unmeasured tape
        is not a calm one, and the verdict below refuses rather than
        assuming.
        """
        if self.realized_percent is None:
            return None
        return implied_move(
            spot=self.spot, iv_percent=self.realized_percent, days=1.0
        )

    @property
    def carry_ratio(self) -> float | None:
        """Breakeven divided by the implied move.

        1.000 for a driftless option — that is the identity. Above it when
        the forward carries a premium, because theta then holds terms the
        driftless case does not: for a call it gains
        `-r*K*exp(-rT)*N(d2) + q*S*exp(-qT)*N(d1)`, and a strong forward
        premium is a large negative `q`.

        Measured live on 4 Sep 2026: NIFTY's forward stood 44 points above
        spot on a 4.2-day expiry, an implied carry of -9.3%, and the ratio
        came out 1.232. So the real bar was 151 points against an implied
        move of 122 — a 24% higher hurdle than the identity alone suggests.
        Worth reporting rather than assuming away, since it moves in the
        direction that makes long gamma harder.
        """
        implied = self.implied_daily_move
        base = self.frictionless_breakeven
        if base is None or implied <= 0:
            return None
        return base / implied

    @property
    def breakeven_with_costs(self) -> float | None:
        if self.drag is None:
            return self.frictionless_breakeven
        return self.drag.move_needed(
            gamma=self.gamma, theta=self.theta, units=self.units
        )

    @property
    def cost_penalty(self) -> float | None:
        """Extra daily move the hedging bill demands, in index points."""
        base, loaded = self.frictionless_breakeven, self.breakeven_with_costs
        if base is None or loaded is None:
            return None
        return loaded - base

    @property
    def pays(self) -> bool | None:
        """Whether the tape is delivering more than the position needs.

        None when realized volatility is unmeasured. True is a necessary
        condition and not a sufficient one: it says the average day clears
        the bar, not that any particular day will.
        """
        realized, needed = self.realized_daily_move, self.breakeven_with_costs
        if realized is None or needed is None:
            return None
        return realized > needed

    def describe(self) -> str:
        ratio = self.carry_ratio
        if ratio is not None and abs(ratio - 1.0) < 0.01:
            head = (
                f"Breakeven daily move {self.frictionless_breakeven:.0f} pts, "
                f"which is the implied move {self.implied_daily_move:.0f} — "
                "identically, as the driftless case requires"
            )
        else:
            head = (
                f"Breakeven daily move {self.frictionless_breakeven:.0f} pts "
                f"against an implied move of {self.implied_daily_move:.0f} "
                f"({ratio:.2f}x) — the gap is the forward's carry, which "
                "raises the bar above the driftless identity"
            )
        parts = [head]
        if self.drag is not None and self.cost_penalty is not None:
            parts.append(
                f"hedging {self.drag.hedges_per_day}x/day adds "
                f"{self.cost_penalty:.0f} pts to the bar"
            )
        realized = self.realized_daily_move
        if realized is None:
            parts.append("realized volatility unmeasured, so no verdict")
        else:
            verdict = "clears it" if self.pays else "does not clear it"
            parts.append(f"realized is delivering {realized:.0f} pts and {verdict}")
        return "; ".join(parts)
