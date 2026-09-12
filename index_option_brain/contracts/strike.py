"""Spec §13.

A StrikeCandidate is a fully-specified, executable structure — not a single
option. A spread is two legs, and its economics (max loss, breakeven,
capital) are properties of the combination, so anything that priced legs
independently would misreport risk. Single-leg structures are simply
one-leg candidates.
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from index_option_brain.contracts.enums import OrderSide, StrategyType
from index_option_brain.contracts.instruments import OptionContractSpec


class StrikeLeg(BaseModel):
    model_config = ConfigDict(frozen=True)

    contract: OptionContractSpec
    side: OrderSide
    lots: int
    reference_price: Decimal
    delta: Decimal | None = None
    liquidity_score: float = 0.0


class StrikeCandidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    strategy: StrategyType
    legs: list[StrikeLeg]
    score: float
    net_premium: Decimal
    """Positive for a net debit paid, negative for a net credit received."""
    net_delta: Decimal
    liquidity_score: float
    worst_relative_spread: float
    capital_required: Decimal
    max_loss: Decimal
    max_profit: Decimal | None
    breakeven: list[Decimal] = Field(default_factory=list)
    rationale: str = ""
    breakeven_sigmas: float | None = None
    """Distance from spot to the nearest breakeven, in one-sigma moves.

    The decisive number when buying premium. A long option only pays if the
    index travels past strike plus premium, and expressing that distance in
    sigmas turns "is this far?" into an answerable question: 1.0 is a move the
    market prices as roughly a one-in-three chance, 2.0 is one-in-forty.

    It is the reason the two "expected move" formulas had to be told apart. If
    the straddle figure is used as a one-sigma band, every breakeven here
    looks 20% closer than it is.
    """
    probability_of_profit: float | None = None
    """Rough chance the structure finishes profitable.

    A **zero-drift normal approximation** to the terminal distribution: it
    ignores drift, the volatility smile, early management, and the whole path.
    Directional for a debit (the move has to happen) and containment for a
    credit (it has to not happen), because those are opposite questions.

    Deliberately not fed into position sizing. It is an order-of-magnitude
    sanity check on whether a structure is a lottery ticket, and treating a
    normal approximation as a real probability is how a model starts sizing on
    its own assumptions.
    """
    round_trip_cost: Decimal = Decimal(0)
    """Brokerage, STT, exchange, SEBI, stamp and GST to open *and* close this
    structure at this size.

    Held separately from `max_profit` rather than folded into it, because an
    operator needs to see the decomposition — but every ranking decision uses
    the net figures below. Measured on the live NIFTY chain, this runs about
    ₹100 on a two-leg spread, which is 3.7% of max profit on a near-the-money
    spread and **14.9%** on one sold four strikes further out. Brokerage is a
    flat fee per order, so it does not shrink with the premium: ranking on
    gross reward-to-risk systematically favours exactly the strikes where
    costs do the most damage.
    """

    @property
    def is_credit(self) -> bool:
        return self.net_premium < 0

    @property
    def reward_to_risk(self) -> float | None:
        """Gross, before costs. Kept for display and for comparison with a
        broker's own figures; `net_reward_to_risk` is what gets ranked."""
        if self.max_profit is None or self.max_loss <= 0:
            return None
        return float(self.max_profit / self.max_loss)

    @property
    def net_max_profit(self) -> Decimal | None:
        """Max profit after costs. None when profit is unbounded."""
        if self.max_profit is None:
            return None
        return self.max_profit - self.round_trip_cost

    @property
    def net_max_loss(self) -> Decimal:
        """Max loss including costs.

        Costs are certain and they are paid on the losing trade too, so the
        real worst case is larger than the market's. The Risk Engine sizes
        against this."""
        return self.max_loss + self.round_trip_cost

    @property
    def net_reward_to_risk(self) -> float | None:
        """What the Strategy and Strike engines rank on.

        On the live chain this is the difference between a spread that looks
        acceptable gross and one that needs a 96% win rate to break even."""
        profit = self.net_max_profit
        if profit is None or self.net_max_loss <= 0:
            return None
        return float(profit / self.net_max_loss)

    @property
    def cost_share_of_profit(self) -> float | None:
        """Costs as a fraction of gross max profit — the number that shows
        when a structure is being eaten by its own execution."""
        if self.max_profit is None or self.max_profit <= 0:
            return None
        return float(self.round_trip_cost / self.max_profit)
