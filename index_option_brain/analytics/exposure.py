"""Portfolio-level greeks: what the book is actually exposed to.

The Risk Engine could count positions and margin, and had no idea of total
directional exposure. Three long 24,100 calls at delta 0.45 read as "3
positions" — not as 87.75 index units, which at 24,000 is about 2.1 crore
of notional NIFTY. Position count and margin do not distinguish that from
three far out-of-the-money lottery tickets.

Why a directional system still needs this
-----------------------------------------
Delta hedging to neutral is not the goal here — the mandate is buying
direction, and a delta-neutral long book earns nothing from being right.
The goal is *knowing and capping* the exposure, and the reason it needs
its own module is gamma.

Gamma means a position sized correctly at entry is oversized after a
favourable move. A single 24,100 call at 0.383 delta becomes 0.533 delta
after a 100-point rally: the same contract, 39% more exposure, bought
without a decision. Any limit checked only at entry is therefore checked
at the one moment it is guaranteed to pass. `projected_delta` exists so a
limit can be tested against where the exposure is going, not where it has
been.

Units, and why they are stated everywhere
-----------------------------------------
Delta exposure is reported in **index-equivalent units** and in rupees,
never as a sum of raw per-contract deltas. Two reasons, both of which have
already caused a real bug in this codebase:

* `PositionLeg.quantity` is in **units** while `StrikeLeg.lots` is in
  **lots**. Two sibling contracts, two conventions. Reading one as the
  other is a 65x error on NIFTY, and it is silent — both produce a
  plausible number.
* NIFTY's lot is 65 and BANKNIFTY's is 30, at index levels around 24,000
  and 57,000. Adding their deltas produces a number that describes no
  portfolio. Exposure is therefore aggregated **per underlying** and only
  ever combined in rupees.

Absence
-------
A leg whose greeks could not be computed does not contribute zero — it
makes the aggregate incomplete, and `is_complete` says so. Zero delta is a
hedged position; unknown delta is an unmeasured one, and a limit that
passes because a leg was invisible is worse than no limit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal

from index_option_brain.contracts.enums import OrderSide
from index_option_brain.contracts.instruments import Greeks, OptionContractSpec


@dataclass(frozen=True)
class LegExposure:
    """One leg's greeks scaled to its actual size, signed by side.

    `units` is contract units, not lots — the caller converts, because only
    the caller knows which convention its source used.
    """

    underlying_symbol: str
    units: int
    side: OrderSide
    greeks: Greeks | None

    @property
    def sign(self) -> int:
        return 1 if self.side is OrderSide.BUY else -1

    @property
    def signed_units(self) -> int:
        return self.units * self.sign

    @property
    def delta_units(self) -> float | None:
        """Index-equivalent exposure: delta x units, signed.

        A long 65-unit call at 0.45 delta is +29.25 units of the index — the
        same directional exposure as holding 29.25 units of NIFTY outright.
        """
        if self.greeks is None:
            return None
        return float(self.greeks.delta) * self.signed_units

    @property
    def gamma_units(self) -> float | None:
        """Change in `delta_units` per one index point of spot move."""
        if self.greeks is None:
            return None
        return float(self.greeks.gamma) * self.signed_units

    @property
    def theta_rupees(self) -> float | None:
        """Rupees per calendar day. Negative is a bleed.

        Theta comes out of `analytics.pricing` per calendar day, not per
        trading day, because an option decays over the weekend.
        """
        if self.greeks is None:
            return None
        return float(self.greeks.theta) * self.signed_units

    @property
    def vega_rupees(self) -> float | None:
        """Rupees per one point of implied volatility."""
        if self.greeks is None:
            return None
        return float(self.greeks.vega) * self.signed_units


def leg_exposure(
    *,
    contract: OptionContractSpec,
    side: OrderSide,
    greeks: Greeks | None,
    lots: int | None = None,
    units: int | None = None,
) -> LegExposure:
    """Build a leg exposure from either convention, but not from neither.

    Exactly one of `lots` and `units` must be given. Defaulting either one
    would let a caller that meant lots be silently read as units, which on
    NIFTY is a 65x error that produces a plausible-looking number — the
    failure mode this signature exists to make impossible.
    """
    if (lots is None) == (units is None):
        raise ValueError(
            "Pass exactly one of lots or units. PositionLeg.quantity is in "
            "units and StrikeLeg.lots is in lots; guessing between them is a "
            f"{contract.lot_size}x error on {contract.underlying_symbol}."
        )
    resolved = units if units is not None else (lots or 0) * contract.lot_size
    return LegExposure(
        underlying_symbol=contract.underlying_symbol.upper(),
        units=resolved,
        side=side,
        greeks=greeks,
    )


@dataclass(frozen=True)
class UnderlyingExposure:
    """Net greeks for one underlying, at one spot level.

    Per underlying because lot sizes and index levels differ: NIFTY's lot is
    65 around 24,000 and BANKNIFTY's is 30 around 57,000, so a sum of their
    deltas describes no portfolio that exists.
    """

    underlying_symbol: str
    spot: Decimal
    legs: list[LegExposure] = field(default_factory=list)

    @property
    def measured(self) -> list[LegExposure]:
        return [leg for leg in self.legs if leg.greeks is not None]

    @property
    def unmeasured_legs(self) -> int:
        return len(self.legs) - len(self.measured)

    @property
    def is_complete(self) -> bool:
        """Whether every leg contributed. False makes the totals a floor.

        A limit that passes because a leg was invisible is worse than no
        limit, so callers check this before trusting a comparison.
        """
        return bool(self.legs) and self.unmeasured_legs == 0

    @property
    def delta_units(self) -> float:
        return sum(leg.delta_units or 0.0 for leg in self.measured)

    @property
    def gamma_units(self) -> float:
        return sum(leg.gamma_units or 0.0 for leg in self.measured)

    @property
    def theta_rupees(self) -> float:
        return sum(leg.theta_rupees or 0.0 for leg in self.measured)

    @property
    def vega_rupees(self) -> float:
        return sum(leg.vega_rupees or 0.0 for leg in self.measured)

    @property
    def delta_notional(self) -> float:
        """Rupee value of the directional exposure: delta units x spot.

        This is the number to compare against capital. "Net delta 500" means
        nothing without the index level behind it.
        """
        return self.delta_units * float(self.spot)

    def projected_delta_units(self, spot_change: float) -> float:
        """Delta after a `spot_change` move, to first order in gamma.

        The point of the module. A long book's delta grows into a favourable
        move, so a position within its limit at entry can breach it while the
        thesis is working — and nobody placed that extra exposure.

        First order only, and that is honest rather than lazy: the next term
        needs the third derivative, and over the fraction of a sigma where a
        limit check matters the gamma term dominates. Over a very large move
        this understates the change.
        """
        return self.delta_units + self.gamma_units * spot_change

    def projected_delta_notional(self, spot_change: float) -> float:
        return self.projected_delta_units(spot_change) * (
            float(self.spot) + spot_change
        )

    def delta_growth(self, spot_change: float) -> float | None:
        """Fractional change in exposure over the move, or None if flat.

        None rather than infinity when current delta is zero: an
        already-neutral book has no growth *ratio*, and reporting a huge
        number there would trip a limit that should not fire.
        """
        if self.delta_units == 0:
            return None
        return (
            self.projected_delta_units(spot_change) - self.delta_units
        ) / abs(self.delta_units)


@dataclass(frozen=True)
class PortfolioExposure:
    """The whole book's greeks, per underlying and in rupees.

    Only the rupee figures are summed across underlyings. Delta units,
    gamma and everything else stay per-symbol, for the reason given on
    `UnderlyingExposure`.
    """

    by_underlying: dict[str, UnderlyingExposure] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        return bool(self.by_underlying) and all(
            exposure.is_complete for exposure in self.by_underlying.values()
        )

    @property
    def unmeasured_legs(self) -> int:
        return sum(e.unmeasured_legs for e in self.by_underlying.values())

    @property
    def delta_notional(self) -> float:
        """Net rupee directional exposure across the book.

        Signed and netted: a long NIFTY position and a short BANKNIFTY one
        partially offset here. They are not a hedge — the two indices are
        correlated, not identical — so this is an exposure measure, never a
        claim that the book is protected.
        """
        return sum(e.delta_notional for e in self.by_underlying.values())

    @property
    def gross_delta_notional(self) -> float:
        """Sum of absolute exposures.

        The number that matters for correlated books, where netting flatters:
        two opposing index positions net to little and can still both lose.
        """
        return sum(abs(e.delta_notional) for e in self.by_underlying.values())

    @property
    def theta_rupees(self) -> float:
        return sum(e.theta_rupees for e in self.by_underlying.values())

    @property
    def vega_rupees(self) -> float:
        return sum(e.vega_rupees for e in self.by_underlying.values())

    def projected_delta_notional(self, sigmas: float, one_sigma: dict[str, float]) -> float:
        """Projected rupee exposure after a `sigmas` move in each underlying.

        `one_sigma` gives each symbol's one-sigma move in index points, so
        the projection is in units of that market's own volatility rather
        than a flat point count — 100 points is a different event on NIFTY
        than on BANKNIFTY. A symbol absent from the mapping is projected
        unchanged rather than at an assumed volatility.
        """
        total = 0.0
        for symbol, exposure in self.by_underlying.items():
            move = one_sigma.get(symbol, 0.0) * sigmas
            total += exposure.projected_delta_notional(move)
        return total

    def delta_share_of_capital(self, capital: Decimal) -> float | None:
        """Gross exposure as a multiple of capital, or None without capital.

        A multiple rather than a percentage because options give exposure
        far above the premium paid: three lots of a 90-rupee call cost about
        17,500 rupees and carry over 20 lakh of delta notional. Anyone
        reading "1,200%" would assume an error.
        """
        if capital <= 0:
            return None
        return self.gross_delta_notional / float(capital)


def portfolio_exposure(
    legs: Sequence[tuple[LegExposure, Decimal]],
) -> PortfolioExposure:
    """Aggregate leg exposures, each paired with its underlying's spot.

    The spot travels with the leg rather than being looked up, so a stale
    price cannot be substituted for a missing one.
    """
    grouped: dict[str, tuple[Decimal, list[LegExposure]]] = {}
    for leg, spot in legs:
        symbol = leg.underlying_symbol
        if symbol not in grouped:
            grouped[symbol] = (spot, [])
        grouped[symbol][1].append(leg)

    return PortfolioExposure(
        by_underlying={
            symbol: UnderlyingExposure(
                underlying_symbol=symbol, spot=spot, legs=legs_for
            )
            for symbol, (spot, legs_for) in grouped.items()
        }
    )
