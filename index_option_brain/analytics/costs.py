"""Transaction costs for Indian index options.

Why this is not a detail
------------------------
Measured against the live NIFTY chain at 6 days to expiry, a two-leg put
credit spread costs about ₹100 to enter and exit. Against the credit
collected, that is:

| Spread (spot 23,914) | Credit | Round trip | Cost as % of max profit |
| --- | --- | --- | --- |
| 23800/23600 | ₹2,896 | ₹107 | 3.7% |
| 23700/23500 | ₹1,817 | ₹103 | 5.6% |
| 23600/23400 | ₹1,082 | ₹100 | 9.2% |
| 23500/23300 | ₹656 | ₹98 | **14.9%** |

The pattern is the point. Brokerage is a **flat fee per order**, so it does
not shrink with the premium — the further out of the money a spread is sold,
the larger a share of its edge the costs take. Ranking structures on
reward-to-risk while ignoring this systematically favours exactly the
strikes where costs do the most damage.

Expressed as breakeven win rates, those same four spreads need 78.5%, 86.8%,
92.4% and 95.7% to break even after costs. A model that cannot see the
difference between the first and the last is not modelling the trade.

The schedule
------------
Rates change by budget and by circular, so every one is a field rather than a
literal. Verify them against the current schedule and your own broker's
contract note before sizing anything real — these are defaults, not
authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

from index_option_brain.contracts.enums import OrderSide

_PAISA = Decimal("0.01")


@dataclass(frozen=True)
class IndianOptionCostModel:
    """Charges for one index-option leg, as of the 2024-25 schedule.

    Defaults follow a discount broker's flat-fee structure, which is what
    makes the flat brokerage dominate on small premiums.
    """

    brokerage_per_order: Decimal = Decimal(20)
    """Flat, per order. Not a percentage — that is the whole reason far-OTM
    premium selling is penalised disproportionately."""
    stt_sell_rate: Decimal = Decimal("0.001")
    """0.1% of premium, **sell side only**, raised from 0.0625% in 2024."""
    exchange_txn_rate: Decimal = Decimal("0.0003503")
    """NSE transaction charge on premium turnover."""
    sebi_turnover_rate: Decimal = Decimal("0.000001")
    stamp_duty_rate: Decimal = Decimal("0.00003")
    """Buy side only, and only on the opening leg."""
    gst_rate: Decimal = Decimal("0.18")
    """On brokerage plus exchange and SEBI charges — not on STT or stamp."""

    def leg_cost(
        self, premium_value: Decimal, *, side: OrderSide, is_opening: bool
    ) -> Decimal:
        """Charges for one leg at one premium turnover.

        `premium_value` is premium x lot size x lots — the money that actually
        changes hands, not the notional of the underlying. Indian option
        charges are levied on premium, and using notional would overstate them
        by two orders of magnitude.
        """
        if premium_value <= 0:
            return Decimal(0)

        is_sell = side is OrderSide.SELL
        brokerage = self.brokerage_per_order
        stt = premium_value * self.stt_sell_rate if is_sell else Decimal(0)
        exchange = premium_value * self.exchange_txn_rate
        sebi = premium_value * self.sebi_turnover_rate
        stamp = (
            premium_value * self.stamp_duty_rate
            if (not is_sell and is_opening)
            else Decimal(0)
        )
        gst = (brokerage + exchange + sebi) * self.gst_rate
        total = brokerage + stt + exchange + sebi + stamp + gst
        return total.quantize(_PAISA, rounding=ROUND_HALF_UP)

    def round_trip(self, legs: list[tuple[Decimal, OrderSide]]) -> Decimal:
        """Cost of opening and closing a whole structure.

        Both halves are counted, because a defined-risk spread is closed, not
        abandoned — and a model that charged only the entry would understate
        the cost of every trade by roughly half. The closing leg reverses the
        side, which matters: STT falls on the sell, so a spread pays it on the
        short leg going in and on the long leg coming out.
        """
        total = Decimal(0)
        for premium_value, side in legs:
            reverse = OrderSide.SELL if side is OrderSide.BUY else OrderSide.BUY
            total += self.leg_cost(premium_value, side=side, is_opening=True)
            total += self.leg_cost(premium_value, side=reverse, is_opening=False)
        return total.quantize(_PAISA, rounding=ROUND_HALF_UP)


DEFAULT_COST_MODEL = IndianOptionCostModel()
