"""Replay the decision chain **with a real option chain**, and score P&L.

What this adds over `replay.py`
--------------------------------
`replay.py` measures whether the analysis layer's directional read leads
price. It cannot measure money, because it was written against the belief
that no historical option chain exists — so the Strike Engine never ran and
the Strategy Engine returned NO_TRADE on all 216 of 216 decisions. That
number described the missing data, not the strategy.

`nse_fo_archive.py` supplies the chain NSE actually publishes, so the Strike
Engine can select real strikes at real settlement prices, and a structure's
outcome can be priced at a later session's settlement and charged through the
real cost model. This module is that loop.

The three assumptions that decide whether the output means anything
-------------------------------------------------------------------
**1. Fills at settlement.** Nobody transacts at the settlement price. There is
no spread to cross here, no queue, no impact — a bhavcopy has no book. Every
trade in this replay therefore gets a better fill than it would have got, and
the error runs one way. That makes a **loss conclusive** and a **profit only
suggestive**: a strategy that cannot make money on free fills certainly cannot
on real ones, while one that can has only earned the right to be measured
against real spreads.

**2. One position at a time.** A signal arriving while a structure is open is
counted and skipped rather than stacked. Overlapping positions would multiply
both the edge and the drawdown by an arbitrary factor set by how often the
engine happens to speak, which is not a property of the strategy.

**3. A contract that did not trade on the exit session is not priced at
zero.** It is recorded as unpriceable and excluded from the P&L, and the count
is reported. Zero is a specific, extremely profitable claim when you are short
and a total loss when you are long; inventing it on an untraded strike is how
a backtest manufactures its best trades.

No lookahead
------------
The state handed to the brain on session `i` is built from bars `[0..i]` and
the chain published at the close of session `i`. Exit prices come from session
`j > i` and are never visible to the decision. The chain is fetched per
session from that session's own bhavcopy, so a strike's price cannot leak
backwards.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from index_option_brain.analytics.costs import DEFAULT_COST_MODEL, IndianOptionCostModel
from index_option_brain.backtest.replay import state_from_bars
from index_option_brain.brain.config import OptionsBrainConfig
from index_option_brain.brain.options_brain import DeterministicOptionsBrain
from index_option_brain.brain.pipeline import QuantitativeBrain
from index_option_brain.contracts.enums import OrderSide, StrategyType
from index_option_brain.contracts.instruments import Bar
from index_option_brain.contracts.market_state import OptionsState
from index_option_brain.contracts.strike import StrikeCandidate

PriceMap = Mapping[tuple[date, Decimal, "object"], Decimal]


@dataclass(frozen=True)
class ReplayTrade:
    """One structure, opened and closed at settlement prices."""

    entry_day: date
    exit_day: date
    strategy: StrategyType
    expiry: date
    legs: int
    entry_premium: Decimal
    """Net debit paid (positive) or credit received (negative)."""
    gross: Decimal
    costs: Decimal
    exit_reason: str

    @property
    def net(self) -> Decimal:
        return self.gross - self.costs


@dataclass
class ChainReplayReport:
    sessions: int = 0
    decisions: int = 0
    tradeable: int = 0
    """Sessions where the Strategy Engine chose a structure and strikes existed."""
    skipped_position_open: int = 0
    unpriceable_exits: int = 0
    trades: list[ReplayTrade] = field(default_factory=list)
    strategies: dict[str, int] = field(default_factory=dict)

    @property
    def gross(self) -> Decimal:
        return sum((t.gross for t in self.trades), Decimal(0))

    @property
    def costs(self) -> Decimal:
        return sum((t.costs for t in self.trades), Decimal(0))

    @property
    def net(self) -> Decimal:
        return self.gross - self.costs

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.net > 0)

    @property
    def hit_rate(self) -> float | None:
        return self.wins / len(self.trades) if self.trades else None

    @property
    def cost_share_of_gross(self) -> float | None:
        """Costs as a fraction of |gross|.

        The number that usually decides an Indian options strategy. A gross
        edge that is real and a cost line that eats 120% of it is a losing
        strategy, and only this ratio makes that visible before the net does.
        """
        gross = abs(self.gross)
        return float(self.costs / gross) if gross > 0 else None


def _leg_value(price: Decimal, *, lot_size: int, lots: int) -> Decimal:
    return price * Decimal(lot_size) * Decimal(lots)


def score_trade(
    candidate: StrikeCandidate,
    *,
    entry_day: date,
    exit_day: date,
    exit_prices: Mapping[tuple[date, Decimal, object], Decimal],
    exit_reason: str,
    cost_model: IndianOptionCostModel = DEFAULT_COST_MODEL,
) -> ReplayTrade | None:
    """P&L for one structure, or `None` when a leg cannot be priced at exit.

    Costs are charged per leg at the premium that actually changed hands on
    each side — entry premium going in, exit premium coming out — rather than
    through `round_trip`, which necessarily assumes one premium for both. On a
    long option that expires worthless the difference is the whole exit cost.
    """
    gross = Decimal(0)
    costs = Decimal(0)
    for leg in candidate.legs:
        spec = leg.contract
        key = (spec.expiry, spec.strike, spec.option_type)
        exit_price = exit_prices.get(key)
        if exit_price is None:
            return None
        entry_value = _leg_value(leg.reference_price, lot_size=spec.lot_size, lots=leg.lots)
        exit_value = _leg_value(exit_price, lot_size=spec.lot_size, lots=leg.lots)
        if leg.side is OrderSide.BUY:
            gross += exit_value - entry_value
            costs += cost_model.leg_cost(entry_value, side=OrderSide.BUY, is_opening=True)
            costs += cost_model.leg_cost(exit_value, side=OrderSide.SELL, is_opening=False)
        else:
            gross += entry_value - exit_value
            costs += cost_model.leg_cost(entry_value, side=OrderSide.SELL, is_opening=True)
            costs += cost_model.leg_cost(exit_value, side=OrderSide.BUY, is_opening=False)
    return ReplayTrade(
        entry_day=entry_day,
        exit_day=exit_day,
        strategy=candidate.strategy,
        expiry=candidate.legs[0].contract.expiry,
        legs=len(candidate.legs),
        entry_premium=candidate.net_premium,
        gross=gross,
        costs=costs,
        exit_reason=exit_reason,
    )


class ChainReplayEngine:
    """Replay index bars plus published chains, and price what it traded."""

    def __init__(
        self,
        *,
        chains: Mapping[date, OptionsState],
        prices: Mapping[date, Mapping[tuple[date, Decimal, object], Decimal]],
        brain: QuantitativeBrain | None = None,
        warmup: int = 30,
        hold_sessions: int = 5,
        cost_model: IndianOptionCostModel = DEFAULT_COST_MODEL,
    ) -> None:
        self._chains = chains
        self._prices = prices
        self._brain = brain or QuantitativeBrain(
            # A bhavcopy has no book, and the Options brain scores an unquoted
            # chain 0.00 by default -- correct for a live feed, where no book
            # means the feed is broken. Over EOD history it vetoed every trade
            # in 90 of 90 decisions, which read as the strategy declining and
            # was the data shape. The fallback is opted into here and nowhere
            # else; live configuration leaves it off.
            options_brain=DeterministicOptionsBrain(
                OptionsBrainConfig(allow_traded_liquidity_fallback=True)
            )
        )
        self._warmup = warmup
        self._hold = hold_sessions
        self._costs = cost_model

    def run(
        self,
        index_symbol: str,
        bars: Sequence[Bar],
        *,
        vix_bars: Sequence[Bar] | None = None,
    ) -> ChainReplayReport:
        if vix_bars is not None and len(vix_bars) != len(bars):
            raise ValueError(
                f"VIX series has {len(vix_bars)} bars against {len(bars)} index bars"
            )
        vix_range: tuple[float, float] | None = None
        if vix_bars:
            closes = [float(b.close) for b in vix_bars]
            vix_range = (max(closes), min(closes))

        report = ChainReplayReport()
        open_until: int | None = None

        for index in range(self._warmup, len(bars)):
            day = bars[index].timestamp.date()
            report.sessions += 1
            chain = self._chains.get(day)
            if chain is None:
                continue

            if open_until is not None and index < open_until:
                report.skipped_position_open += 1
                continue

            state = state_from_bars(
                index_symbol=index_symbol,
                bars=bars[: index + 1],
                vix_bars=vix_bars[: index + 1] if vix_bars else None,
                vix_year_range=vix_range,
            ).model_copy(update={"options_state": chain})
            result = self._brain.run(state)
            if result.regime is None:
                continue
            report.decisions += 1
            name = str(result.selected_strategy)
            report.strategies[name] = report.strategies.get(name, 0) + 1
            if result.selected_strategy is StrategyType.NO_TRADE:
                continue
            if not result.strike_candidates:
                continue
            candidate = result.strike_candidates[0]
            report.tradeable += 1

            exit_index, reason = self._exit_index(bars, index, candidate)
            if exit_index is None:
                continue
            exit_day = bars[exit_index].timestamp.date()
            exit_prices = self._prices.get(exit_day)
            if exit_prices is None:
                report.unpriceable_exits += 1
                continue
            trade = score_trade(
                candidate,
                entry_day=day,
                exit_day=exit_day,
                exit_prices=exit_prices,
                exit_reason=reason,
                cost_model=self._costs,
            )
            if trade is None:
                report.unpriceable_exits += 1
                continue
            report.trades.append(trade)
            open_until = exit_index

        return report

    def _exit_index(
        self, bars: Sequence[Bar], index: int, candidate: StrikeCandidate
    ) -> tuple[int | None, str]:
        """Hold `hold_sessions`, or to expiry, whichever comes first.

        Expiry wins because a contract held past it does not exist, and the
        bhavcopy's expiry-day settlement *is* the intrinsic value — so exiting
        there is the same cash flow as being settled.
        """
        expiry = candidate.legs[0].contract.expiry
        horizon = min(index + self._hold, len(bars) - 1)
        for offset in range(index + 1, horizon + 1):
            if bars[offset].timestamp.date() >= expiry:
                return offset, "expiry"
        if horizon <= index:
            return None, ""
        return horizon, "horizon"
