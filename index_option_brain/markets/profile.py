"""What differs between venues, isolated so the brain does not have to care.

The decision layer is market-agnostic and always was: a delta floor, a
regime classification, an implied-versus-realized comparison and a
portfolio exposure limit are statements about options, not about India.
What is *not* portable sits in four places, and this module is those four
places made explicit rather than assumed:

* **When the market is open.** NSE runs 09:15-15:30 IST with a pre-open
  auction; a crypto venue runs continuously. Every session-derived
  measurement — the opening range, breadth from the auction, whether a
  snapshot counts as live — depends on which.
* **What a trade costs.** India charges a flat 20 rupees per order plus
  STT, exchange, SEBI, stamp and 18% GST. Delta Exchange charges a
  commission rate on notional. These are not the same shape of formula,
  so the cost model is an object, not a set of constants.
* **What one contract is.** NIFTY's lot is 65 units of the index; a Delta
  BTC option's `contract_value` is 0.001 BTC. Both are read from the
  venue rather than hardcoded — the one time this codebase hardcoded a
  lot size it was wrong by 15%.
* **What it settles in.** Rupees against USD. Every rupee figure in the
  risk limits is a currency-bearing number, and mixing them silently is
  the failure this makes visible.

What deliberately does *not* live here: anything the brain reasons with.
Adding a market must not require touching a brain, and if it ever does,
that is a sign the brain has absorbed a venue assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from decimal import Decimal
from enum import StrEnum

from index_option_brain.analytics.costs import IndianOptionCostModel
from index_option_brain.contracts.enums import MarketSessionState, OrderSide

IST = timezone(timedelta(hours=5, minutes=30))


class SessionModel(StrEnum):
    """How a venue's trading day is shaped."""

    SCHEDULED = "SCHEDULED"
    """Fixed daily hours in a local timezone, with an auction and a close."""
    CONTINUOUS = "CONTINUOUS"
    """Always open. There is no close, so nothing derived from one exists."""


@dataclass(frozen=True)
class NotionalCommissionModel:
    """A commission charged as a rate on traded notional.

    Delta Exchange publishes `maker_commission_rate` and
    `taker_commission_rate` on every product — 0.0001 on BTC options at the
    time of writing. The taker rate is used because a system that reads a
    book and then trades against it is taking liquidity; assuming the maker
    rate would understate every cost the moment an order actually fills.

    Deliberately a different type from `IndianOptionCostModel` rather than a
    parameterisation of it. India's charge is a flat fee plus five
    percentage levies plus GST on a subset of them; this is one
    multiplication. Forcing both through one signature would mean a formula
    with fields that are zero for whichever venue is not in use, and a zero
    there is indistinguishable from a levy someone forgot to configure.
    """

    taker_rate: Decimal = Decimal("0.0001")
    settlement_rate: Decimal = Decimal("0.0001")
    """Charged when an in-the-money option settles rather than being closed.

    Kept separate because it applies to exactly one leg of a round trip and
    only when the option expires with value, which a symmetric per-side
    rate would model wrongly.
    """

    def leg_cost(self, *, notional: Decimal, side: OrderSide) -> Decimal:
        """Commission on one leg. Side is accepted but does not change it.

        Both sides pay the same rate here, unlike India where STT falls only
        on the sell. The parameter is kept so the two cost models share a
        call shape and a caller cannot silently drop the distinction when
        moving between venues.
        """
        del side
        return abs(notional) * self.taker_rate

    def round_trip(self, legs: list[tuple[Decimal, OrderSide]]) -> Decimal:
        """Cost of opening and closing every leg."""
        return sum(
            (self.leg_cost(notional=notional, side=side) for notional, side in legs),
            Decimal(0),
        ) * 2


@dataclass(frozen=True)
class MarketProfile:
    """One venue's non-portable facts.

    `contract_multiplier_field` names where the venue publishes the size of
    one contract, so the adapter reads it rather than a constant. It is a
    string because the point is that the value is fetched, not that it is
    known here.
    """

    key: str
    name: str
    session_model: SessionModel
    currency: str
    cost_model: IndianOptionCostModel | NotionalCommissionModel
    timezone: timezone | None = None
    opens_at: time | None = None
    opening_ends_at: time | None = None
    closing_begins_at: time | None = None
    closes_at: time | None = None
    contract_multiplier_field: str = ""
    has_constituents: bool = False
    """Whether the underlying decomposes into constituents at all.

    NIFTY has 50; BTC has none. The Constituent brain reports an absence
    either way, and this is what distinguishes "breadth is unavailable right
    now" from "breadth is not a thing in this market" — a distinction that
    matters because the first is a gap worth closing and the second is not.
    """

    def session_state(self, moment: datetime) -> MarketSessionState:
        """Which part of the trading day `moment` falls in.

        A continuous venue is always ACTIVE. Returning CLOSED overnight
        there would make every session-derived measurement silently switch
        off for hours at a time.
        """
        if self.session_model is SessionModel.CONTINUOUS:
            return MarketSessionState.ACTIVE
        if self.timezone is None or self.opens_at is None or self.closes_at is None:
            raise ValueError(
                f"{self.key} is SCHEDULED but has no hours configured, so no "
                "session state can be derived"
            )
        local = moment.astimezone(self.timezone).time()
        if local < self.opens_at:
            return MarketSessionState.PRE_MARKET
        if self.opening_ends_at is not None and local < self.opening_ends_at:
            return MarketSessionState.OPENING
        if self.closing_begins_at is not None and local < self.closing_begins_at:
            return MarketSessionState.ACTIVE
        if local < self.closes_at:
            return MarketSessionState.CLOSING
        return MarketSessionState.CLOSED

    @property
    def has_opening_range(self) -> bool:
        """Whether an opening range is a meaningful construct here.

        It needs an open. A continuous market has none, and computing one
        from an arbitrary UTC midnight would produce a level that looks like
        structure and is an artifact of the clock.
        """
        return self.session_model is SessionModel.SCHEDULED


NSE_INDIA = MarketProfile(
    key="nse_india",
    name="NSE India index options",
    session_model=SessionModel.SCHEDULED,
    currency="INR",
    cost_model=IndianOptionCostModel(),
    timezone=IST,
    opens_at=time(9, 15),
    opening_ends_at=time(9, 30),
    closing_begins_at=time(15, 0),
    closes_at=time(15, 30),
    contract_multiplier_field="lot_size",
    has_constituents=True,
)

CRYPTO_24_7 = MarketProfile(
    key="delta_india",
    name="Delta Exchange India crypto options",
    session_model=SessionModel.CONTINUOUS,
    currency="USD",
    cost_model=NotionalCommissionModel(),
    contract_multiplier_field="contract_value",
    has_constituents=False,
)
