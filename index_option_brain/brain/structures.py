"""Construction and pricing of option structures.

Shared by the Strategy Engine (which needs indicative economics to compare
structures) and the Strike Engine (which needs actual executable legs). One
implementation, so the risk numbers a strategy was chosen on are the same
numbers the ranked contracts report — a mismatch between the two is how a
system ends up authorizing a trade whose real max loss nobody computed.

Pricing is deliberately pessimistic: buys are priced at the ask and sells at
the bid, falling back to LTP only when a side is unquoted. Mid-pricing a
spread flatters every number in this module, and the flattery compounds
through the Risk Engine.

All money values are per-position totals in rupees (premium x lot size x
lots), not per-unit premiums.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from itertools import pairwise

from index_option_brain.analytics.costs import (
    DEFAULT_COST_MODEL,
    IndianOptionCostModel,
)
from index_option_brain.contracts.enums import OptionType, OrderSide, StrategyType
from index_option_brain.contracts.instruments import OptionQuote
from index_option_brain.contracts.market_state import MarketState
from index_option_brain.contracts.strike import StrikeCandidate, StrikeLeg

_UNLIMITED = None


@dataclass(frozen=True)
class ChainView:
    """An option chain indexed for structure construction."""

    by_strike: dict[Decimal, dict[OptionType, OptionQuote]]
    strikes: list[Decimal]
    spot: Decimal
    atm_index: int
    step: Decimal
    lot_size: int

    @classmethod
    def from_state(cls, state: MarketState) -> ChainView | None:
        return cls.from_chain(state.options_state.chain, state.spot, state)

    @classmethod
    def from_chain(
        cls, chain: list[OptionQuote], spot: Decimal, state: MarketState | None = None
    ) -> ChainView | None:
        if not chain:
            return None

        by_strike: dict[Decimal, dict[OptionType, OptionQuote]] = {}
        for quote in chain:
            by_strike.setdefault(quote.contract.strike, {})[quote.contract.option_type] = quote

        strikes = sorted(by_strike)
        if len(strikes) < 2:
            return None

        atm_strike = min(strikes, key=lambda s: abs(s - spot))
        atm_index = strikes.index(atm_strike)

        gaps = [b - a for a, b in pairwise(strikes) if b > a]
        step = sorted(gaps)[len(gaps) // 2] if gaps else Decimal(50)

        lot_size = chain[0].contract.lot_size
        if state is not None and state.index_state.spec is not None:
            lot_size = state.index_state.spec.lot_size

        return cls(
            by_strike=by_strike,
            strikes=strikes,
            spot=spot,
            atm_index=atm_index,
            step=step,
            lot_size=lot_size,
        )

    @property
    def atm_strike(self) -> Decimal:
        return self.strikes[self.atm_index]

    def strike_at(self, offset: int) -> Decimal | None:
        """Strike `offset` steps from ATM; positive is higher."""
        index = self.atm_index + offset
        if index < 0 or index >= len(self.strikes):
            return None
        return self.strikes[index]

    def quote(self, strike: Decimal | None, option_type: OptionType) -> OptionQuote | None:
        if strike is None:
            return None
        return self.by_strike.get(strike, {}).get(option_type)


def execution_price(quote: OptionQuote, side: OrderSide) -> Decimal:
    """What this side would realistically pay/receive, not the mid."""
    if side is OrderSide.BUY:
        if quote.ask is not None and quote.ask > 0:
            return quote.ask
    elif quote.bid is not None and quote.bid > 0:
        return quote.bid
    return quote.ltp


def leg_liquidity(quote: OptionQuote, max_relative_spread: float) -> float:
    relative = quote.relative_spread
    if relative is None:
        return 0.0
    if max_relative_spread <= 0:
        return 0.0
    score = 1.0 - float(relative) / max_relative_spread
    return max(0.0, min(1.0, score))


def _make_leg(
    quote: OptionQuote, side: OrderSide, lots: int, max_relative_spread: float
) -> StrikeLeg:
    return StrikeLeg(
        contract=quote.contract,
        side=side,
        lots=lots,
        reference_price=execution_price(quote, side),
        delta=quote.greeks.delta if quote.greeks is not None else None,
        liquidity_score=leg_liquidity(quote, max_relative_spread),
    )


def _economics(
    strategy: StrategyType,
    legs: list[StrikeLeg],
    view: ChainView,
    lots: int,
) -> tuple[Decimal, Decimal, Decimal | None, list[Decimal]] | None:
    """Return (net_premium, max_loss, max_profit, breakevens).

    net_premium is positive for a net debit paid and negative for a net
    credit received.
    """
    multiplier = Decimal(view.lot_size * lots)
    net_premium = sum(
        (
            leg.reference_price * Decimal(leg.lots * view.lot_size)
            * (Decimal(1) if leg.side is OrderSide.BUY else Decimal(-1))
            for leg in legs
        ),
        Decimal(0),
    )
    per_unit = net_premium / multiplier if multiplier else Decimal(0)

    if strategy in (StrategyType.LONG_CALL, StrategyType.LONG_PUT):
        if net_premium <= 0:
            return None
        strike = legs[0].contract.strike
        breakeven = (
            strike + per_unit if strategy is StrategyType.LONG_CALL else strike - per_unit
        )
        return net_premium, net_premium, _UNLIMITED, [breakeven]

    if strategy in (StrategyType.CALL_DEBIT_SPREAD, StrategyType.PUT_DEBIT_SPREAD):
        if net_premium <= 0:
            return None
        long_leg = next(leg for leg in legs if leg.side is OrderSide.BUY)
        short_leg = next(leg for leg in legs if leg.side is OrderSide.SELL)
        width = abs(short_leg.contract.strike - long_leg.contract.strike)
        max_profit = width * multiplier - net_premium
        if max_profit <= 0:
            return None
        breakeven = (
            long_leg.contract.strike + per_unit
            if strategy is StrategyType.CALL_DEBIT_SPREAD
            else long_leg.contract.strike - per_unit
        )
        return net_premium, net_premium, max_profit, [breakeven]

    if strategy in (StrategyType.CALL_CREDIT_SPREAD, StrategyType.PUT_CREDIT_SPREAD):
        if net_premium >= 0:
            return None
        credit = -net_premium
        long_leg = next(leg for leg in legs if leg.side is OrderSide.BUY)
        short_leg = next(leg for leg in legs if leg.side is OrderSide.SELL)
        width = abs(long_leg.contract.strike - short_leg.contract.strike)
        max_loss = width * multiplier - credit
        if max_loss <= 0:
            return None
        credit_per_unit = credit / multiplier
        breakeven = (
            short_leg.contract.strike + credit_per_unit
            if strategy is StrategyType.CALL_CREDIT_SPREAD
            else short_leg.contract.strike - credit_per_unit
        )
        return net_premium, max_loss, credit, [breakeven]

    if strategy is StrategyType.NEUTRAL_DEFINED_RISK:
        if net_premium >= 0:
            return None
        credit = -net_premium
        calls = [leg for leg in legs if leg.contract.option_type is OptionType.CE]
        puts = [leg for leg in legs if leg.contract.option_type is OptionType.PE]
        if len(calls) != 2 or len(puts) != 2:
            return None
        short_call = next(leg for leg in calls if leg.side is OrderSide.SELL)
        long_call = next(leg for leg in calls if leg.side is OrderSide.BUY)
        short_put = next(leg for leg in puts if leg.side is OrderSide.SELL)
        long_put = next(leg for leg in puts if leg.side is OrderSide.BUY)
        call_width = long_call.contract.strike - short_call.contract.strike
        put_width = short_put.contract.strike - long_put.contract.strike
        width = max(call_width, put_width)
        max_loss = width * multiplier - credit
        if max_loss <= 0:
            return None
        credit_per_unit = credit / multiplier
        return (
            net_premium,
            max_loss,
            credit,
            [
                short_call.contract.strike + credit_per_unit,
                short_put.contract.strike - credit_per_unit,
            ],
        )

    return None


def build_structure(
    strategy: StrategyType,
    view: ChainView,
    *,
    anchor_offset: int = 0,
    width_steps: int = 2,
    lots: int = 1,
    max_relative_spread: float = 0.08,
    cost_model: IndianOptionCostModel | None = None,
) -> StrikeCandidate | None:
    """Build one executable structure, or None if the chain can't support it.

    `anchor_offset` positions the primary (usually short, or long for debit
    structures) strike relative to ATM in strike steps; `width_steps` is the
    distance to the protective/short wing.
    """
    if strategy is StrategyType.NO_TRADE or lots <= 0 or width_steps <= 0:
        return None

    legs: list[StrikeLeg] = []

    def add(strike: Decimal | None, option_type: OptionType, side: OrderSide) -> bool:
        quote = view.quote(strike, option_type)
        if quote is None or execution_price(quote, side) <= 0:
            return False
        legs.append(_make_leg(quote, side, lots, max_relative_spread))
        return True

    if strategy is StrategyType.LONG_CALL:
        if not add(view.strike_at(anchor_offset), OptionType.CE, OrderSide.BUY):
            return None
    elif strategy is StrategyType.LONG_PUT:
        if not add(view.strike_at(anchor_offset), OptionType.PE, OrderSide.BUY):
            return None
    elif strategy is StrategyType.CALL_DEBIT_SPREAD:
        if not add(view.strike_at(anchor_offset), OptionType.CE, OrderSide.BUY):
            return None
        if not add(view.strike_at(anchor_offset + width_steps), OptionType.CE, OrderSide.SELL):
            return None
    elif strategy is StrategyType.PUT_DEBIT_SPREAD:
        if not add(view.strike_at(anchor_offset), OptionType.PE, OrderSide.BUY):
            return None
        if not add(view.strike_at(anchor_offset - width_steps), OptionType.PE, OrderSide.SELL):
            return None
    elif strategy is StrategyType.CALL_CREDIT_SPREAD:
        if not add(view.strike_at(anchor_offset), OptionType.CE, OrderSide.SELL):
            return None
        if not add(view.strike_at(anchor_offset + width_steps), OptionType.CE, OrderSide.BUY):
            return None
    elif strategy is StrategyType.PUT_CREDIT_SPREAD:
        if not add(view.strike_at(anchor_offset), OptionType.PE, OrderSide.SELL):
            return None
        if not add(view.strike_at(anchor_offset - width_steps), OptionType.PE, OrderSide.BUY):
            return None
    elif strategy is StrategyType.NEUTRAL_DEFINED_RISK:
        wing = abs(anchor_offset) or 2
        if not add(view.strike_at(wing), OptionType.CE, OrderSide.SELL):
            return None
        if not add(view.strike_at(wing + width_steps), OptionType.CE, OrderSide.BUY):
            return None
        if not add(view.strike_at(-wing), OptionType.PE, OrderSide.SELL):
            return None
        if not add(view.strike_at(-wing - width_steps), OptionType.PE, OrderSide.BUY):
            return None
    else:
        return None

    economics = _economics(strategy, legs, view, lots)
    if economics is None:
        return None
    net_premium, max_loss, max_profit, breakevens = economics

    net_delta = sum(
        (
            (leg.delta or Decimal(0))
            * Decimal(leg.lots * view.lot_size)
            * (Decimal(1) if leg.side is OrderSide.BUY else Decimal(-1))
            for leg in legs
        ),
        Decimal(0),
    )

    liquidity_score = min(leg.liquidity_score for leg in legs)
    worst_relative_spread = 0.0
    for leg in legs:
        quote = view.quote(leg.contract.strike, leg.contract.option_type)
        if quote is not None and quote.relative_spread is not None:
            worst_relative_spread = max(worst_relative_spread, float(quote.relative_spread))

    # Defined-risk structures are collateralized by their own maximum loss;
    # long premium costs only the debit. Real margin comes from the broker and
    # is re-checked by the Risk Engine before anything is authorized.
    capital_required = max_loss if net_premium < 0 else net_premium

    # Costs are computed from the legs as priced, at this size. Charges fall
    # on premium turnover, not on the notional of the underlying — using
    # notional would overstate them by two orders of magnitude.
    model = cost_model or DEFAULT_COST_MODEL
    round_trip_cost = model.round_trip(
        [
            (
                leg.reference_price * leg.contract.lot_size * leg.lots,
                leg.side,
            )
            for leg in legs
        ]
    )

    return StrikeCandidate(
        strategy=strategy,
        legs=legs,
        score=0.0,
        net_premium=net_premium,
        net_delta=net_delta,
        liquidity_score=liquidity_score,
        worst_relative_spread=worst_relative_spread,
        capital_required=capital_required,
        max_loss=max_loss,
        max_profit=max_profit,
        breakeven=breakevens,
        rationale="",
        round_trip_cost=round_trip_cost,
    )
