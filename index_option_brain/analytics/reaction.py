"""What an option does when the index moves — and why.

The greeks are rates of change, and a trader needs the change. This turns
"delta 0.56, gamma 0.0014, theta -11.4" into "if NIFTY opens 150 up, this
call goes from 131.60 to 227, and here is which greek did what".

Exact, then decomposed
----------------------
Two numbers are produced for every scenario and they are not the same:

* the **exact** new price, from repricing with Black-Scholes at the new spot,
  time and IV — this is what the option is actually worth;
* the **greek estimate**, from the Taylor expansion
  `delta*dS + 0.5*gamma*dS^2 + theta*dt + vega*dIV`.

Keeping both is the point. The gap between them is the error in the
approximation every desk uses in its head, and it grows exactly where it
matters: large moves, near expiry, and short gamma. A projection that only
showed the estimate would be most confident precisely where it is most wrong.

Why the decomposition matters more than the total
-------------------------------------------------
Two calls can both gain 40 points and be entirely different trades — one
because the index moved and one because volatility repriced. On a long
option, theta is the standing cost of being wrong slowly, and seeing it as a
separate line is what stops "it moved my way and I still lost" being a
surprise.
"""

from __future__ import annotations

from dataclasses import dataclass

from index_option_brain.analytics.pricing import (
    DEFAULT_RISK_FREE_RATE,
    price_option,
)
from index_option_brain.contracts.enums import OptionType


@dataclass(frozen=True)
class Scenario:
    """One what-if: a move in the index, time passing, volatility repricing."""

    label: str
    spot_change: float = 0.0
    """In index points."""
    days_elapsed: float = 0.0
    """Calendar days. An overnight gap is one."""
    iv_change: float = 0.0
    """In IV points, so +2.0 means 10% IV becomes 12%."""


@dataclass(frozen=True)
class Reaction:
    """What one option does under one scenario."""

    scenario: Scenario
    price_before: float
    price_after: float
    delta_contribution: float
    gamma_contribution: float
    theta_contribution: float
    vega_contribution: float
    delta_before: float
    delta_after: float

    @property
    def change(self) -> float:
        return self.price_after - self.price_before

    @property
    def change_pct(self) -> float | None:
        if self.price_before <= 0:
            return None
        return self.change / self.price_before * 100.0

    @property
    def greek_estimate(self) -> float:
        """What the Taylor expansion predicts the change will be."""
        return (
            self.delta_contribution
            + self.gamma_contribution
            + self.theta_contribution
            + self.vega_contribution
        )

    @property
    def approximation_error(self) -> float:
        """Exact change minus the greek estimate.

        Large where the approximation is worst — big moves, near expiry — and
        the reason both numbers are kept rather than just the tidy one.
        """
        return self.change - self.greek_estimate

    def per_lot(self, lot_size: int, lots: int = 1) -> float:
        """The change in rupees for a real position."""
        return self.change * lot_size * lots


def project(
    *,
    spot: float,
    strike: float,
    years: float,
    iv_percent: float,
    option_type: OptionType,
    scenario: Scenario,
    rate: float = DEFAULT_RISK_FREE_RATE,
) -> Reaction:
    """Reprice one option under one scenario and decompose the move.

    Time is floored at expiry rather than allowed to go negative: a scenario
    that runs past expiry values the option at intrinsic, which is what
    actually happens, instead of producing a mathematical artifact.
    """
    before = price_option(
        spot=spot,
        strike=strike,
        years=years,
        iv=iv_percent / 100.0,
        option_type=option_type,
        rate=rate,
    )

    new_spot = spot + scenario.spot_change
    new_years = max(0.0, years - scenario.days_elapsed / 365.0)
    new_iv = max(0.0, iv_percent + scenario.iv_change)
    after = price_option(
        spot=new_spot,
        strike=strike,
        years=new_years,
        iv=new_iv / 100.0,
        option_type=option_type,
        rate=rate,
    )

    move = scenario.spot_change
    return Reaction(
        scenario=scenario,
        price_before=before.price,
        price_after=after.price,
        # Delta is per point, gamma is per point squared and enters with the
        # one-half from the second-order term, theta is per calendar day, and
        # vega is per IV point — the four units this codebase quotes them in.
        delta_contribution=before.delta * move,
        gamma_contribution=0.5 * before.gamma * move * move,
        theta_contribution=before.theta * scenario.days_elapsed,
        vega_contribution=before.vega * scenario.iv_change,
        delta_before=before.delta,
        delta_after=after.delta,
    )


def gap_scenarios(
    one_day_sigma: float, *, overnight: bool = True
) -> list[Scenario]:
    """A ladder of opening gaps, in sigmas rather than points.

    Sigmas because points mean nothing without the volatility that produced
    them: 150 points is a routine morning at 20% IV and a violent one at 10%.
    One calendar day of decay is included by default, since an overnight gap
    always costs a day whichever way the index goes — the thing option buyers
    most often forget to price.
    """
    days = 1.0 if overnight else 0.0
    ladder: list[Scenario] = []
    for multiple in (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0):
        move = one_day_sigma * multiple
        if multiple == 0.0:
            label = "flat open"
        else:
            label = f"{multiple:+.1f} sigma ({move:+.0f} pts)"
        ladder.append(
            Scenario(label=label, spot_change=move, days_elapsed=days)
        )
    return ladder


def volatility_crush(points: float = -1.5, days: float = 1.0) -> Scenario:
    """The move that hurts a buyer without the index doing anything.

    A quiet open after a nervous close reprices volatility down, and a long
    option pays for it twice — once through vega and once through theta. It
    is the single most common way a directionally correct buyer still loses.
    """
    return Scenario(
        label=f"flat open, IV {points:+.1f} pts",
        spot_change=0.0,
        days_elapsed=days,
        iv_change=points,
    )
