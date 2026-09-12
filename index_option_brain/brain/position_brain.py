"""Spec §18. Core question this brain answers on every tick for every open
position: "Is the reason for entering the trade still valid?"

Note what that question is *not*: it is not "am I in profit". A position can
be green while its thesis is dead (the move happened for an unrelated reason
and is about to mean-revert) and red while its thesis is intact (the entry
was early). So the checks here are ordered by cause, not by P&L:

  1. Has the thesis broken — does current index analysis now oppose the
     direction the trade was entered on?
  2. Has risk been exceeded — is loss approaching the max loss the trade was
     authorized for?
  3. Has the objective been met — is the target reached?
  4. Has the market stopped being able to exit us — has liquidity collapsed,
     or is expiry too close to manage?

This brain only re-states the position and its evidence. It never places or
modifies an order; the returned lifecycle state is a *request* that the
Execution Gate and Order Manager act on (spec §16, §17).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal

from index_option_brain.brain.config import PositionBrainConfig
from index_option_brain.contracts.enums import Direction, OrderSide, TradeLifecycleState
from index_option_brain.contracts.market_state import MarketState
from index_option_brain.contracts.position import Position, PositionLeg


class PositionBrain(ABC):
    @abstractmethod
    def evaluate(self, position: Position, state: MarketState) -> Position:
        """Return an updated Position (state transition, refreshed P&L, and
        thesis-validity evidence). Must never place or modify a broker order
        directly — that is the Execution Gate / Order Manager's job."""
        ...


class DeterministicPositionBrain(PositionBrain):
    def __init__(self, config: PositionBrainConfig | None = None) -> None:
        self._config = config or PositionBrainConfig()

    def evaluate(self, position: Position, state: MarketState) -> Position:
        if not position.is_open:
            return position

        cfg = self._config
        legs = self._reprice(position, state)
        unrealized = sum((leg.unrealized_pnl() for leg in legs), Decimal(0))

        evidence: list[str] = []
        next_state = position.state
        exit_reasons: list[str] = []

        thesis_broken, thesis_note = self._thesis_broken(position, state, cfg)
        if thesis_note:
            evidence.append(thesis_note)
        if thesis_broken:
            exit_reasons.append("thesis invalidated")

        if position.max_loss > 0:
            loss_fraction = float(-unrealized / position.max_loss) if unrealized < 0 else 0.0
            if loss_fraction >= cfg.stop_fraction_of_max_loss:
                exit_reasons.append(
                    f"loss at {loss_fraction * 100:.0f}% of authorized max loss"
                )
            elif loss_fraction > 0:
                evidence.append(
                    f"Drawdown is {loss_fraction * 100:.0f}% of authorized max loss"
                )

        if position.target_profit is not None and unrealized >= position.target_profit:
            exit_reasons.append(
                f"target of {position.target_profit} reached (P&L {unrealized:.0f})"
            )

        liquidity_note = self._liquidity_failure(legs, state, cfg)
        if liquidity_note is not None:
            exit_reasons.append(liquidity_note)

        days = state.volatility_state.days_to_expiry
        if days is not None and days <= cfg.min_days_to_expiry:
            exit_reasons.append(f"{days:.2f} days to expiry — too close to manage")

        if exit_reasons:
            next_state = TradeLifecycleState.EXIT_PENDING
            evidence.append("Exit requested: " + "; ".join(exit_reasons))
        elif position.state is TradeLifecycleState.ACTIVE and thesis_broken is False and (
            thesis_note is not None and "weakening" in thesis_note
        ):
            # Thesis is intact but deteriorating: escalate to explicit testing
            # rather than waiting for it to break.
            next_state = TradeLifecycleState.THESIS_TEST
            evidence.append("Thesis is weakening — monitoring under test")
        elif position.state is TradeLifecycleState.THESIS_TEST and not thesis_broken:
            next_state = TradeLifecycleState.ACTIVE
            evidence.append("Thesis re-confirmed")

        return position.model_copy(
            update={
                "legs": legs,
                "unrealized_pnl": unrealized,
                "state": next_state,
                "evidence": evidence,
                # Stamped from the market state, never the wall clock: the
                # same brain runs in BACKTEST and REPLAY (spec §22), where
                # `now()` would date historical positions to today and make
                # the audit trail non-reproducible.
                "updated_at": state.timestamp,
            }
        )

    def _reprice(self, position: Position, state: MarketState) -> list[PositionLeg]:
        """Mark each leg at what it could actually be closed for: a long is
        sold into the bid, a short is bought back at the ask."""
        chain = {
            (q.contract.strike, q.contract.option_type, q.contract.expiry): q
            for q in state.options_state.chain
        }
        out: list[PositionLeg] = []
        for leg in position.legs:
            key = (leg.contract.strike, leg.contract.option_type, leg.contract.expiry)
            quote = chain.get(key)
            if quote is None:
                out.append(leg)
                continue
            if leg.side is OrderSide.BUY:
                price = quote.bid if quote.bid is not None and quote.bid > 0 else quote.ltp
            else:
                price = quote.ask if quote.ask is not None and quote.ask > 0 else quote.ltp
            out.append(leg.model_copy(update={"current_price": price}))
        return out

    def _thesis_broken(
        self, position: Position, state: MarketState, cfg: PositionBrainConfig
    ) -> tuple[bool, str | None]:
        analysis = state.analysis
        if analysis is None:
            return False, "No current analysis — thesis validity cannot be reassessed"

        composite = analysis.index.composite_score
        if position.thesis_direction is Direction.BULLISH:
            aligned = composite
        elif position.thesis_direction is Direction.BEARISH:
            aligned = -composite
        else:
            # A neutral thesis breaks on movement, not on direction.
            aligned = -abs(composite)

        if aligned <= cfg.thesis_break_score:
            return True, (
                f"Index composite {composite:+.2f} now opposes the "
                f"{position.thesis_direction.value} thesis"
            )
        if aligned < 0:
            return False, (
                f"Index composite {composite:+.2f} is weakening against the "
                f"{position.thesis_direction.value} thesis"
            )
        return False, f"Index composite {composite:+.2f} still supports the thesis"

    def _liquidity_failure(
        self, legs: list[PositionLeg], state: MarketState, cfg: PositionBrainConfig
    ) -> str | None:
        """A position that cannot be exited is a different risk from a
        position that is losing."""
        chain = {
            (q.contract.strike, q.contract.option_type, q.contract.expiry): q
            for q in state.options_state.chain
        }
        for leg in legs:
            key = (leg.contract.strike, leg.contract.option_type, leg.contract.expiry)
            quote = chain.get(key)
            if quote is None:
                continue
            relative = quote.relative_spread
            if relative is not None and float(relative) > cfg.liquidity_exit_relative_spread:
                return (
                    f"liquidity deteriorated on {leg.contract.strike:.0f}"
                    f"{leg.contract.option_type.value} "
                    f"(spread {float(relative) * 100:.1f}%)"
                )
        return None
