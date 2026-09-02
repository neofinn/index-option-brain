"""Spec §13.

Enumerates every viable placement of the chosen structure across the chain,
prices each one from the live quotes, discards the ones that fail hard
filters, and ranks what survives.

Three things are scored, and a candidate has to be decent at all three:
delta fit (is the exposure the right size for the intent), liquidity (can
this be entered and, more importantly, exited), and structural quality
(reward-to-risk, capital efficiency, and whether the trade is buying
straight into a wall). Ranking on delta alone reliably picks illiquid
strikes; ranking on liquidity alone reliably picks the ATM strike whatever
the thesis.

Hard filters run before scoring, because no score should be able to rescue a
contract that is untradeable — spec §16 will reject it at the gate anyway,
and it is better to never have proposed it.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from decimal import Decimal

from index_option_brain.brain import indicators as ind
from index_option_brain.brain import structures
from index_option_brain.brain.config import StrikeEngineConfig
from index_option_brain.contracts.enums import OptionType, OrderSide, StrategyType
from index_option_brain.contracts.instruments import OptionQuote
from index_option_brain.contracts.market_state import MarketState
from index_option_brain.contracts.strike import StrikeCandidate

_CREDIT_STRATEGIES = {
    StrategyType.CALL_CREDIT_SPREAD,
    StrategyType.PUT_CREDIT_SPREAD,
    StrategyType.NEUTRAL_DEFINED_RISK,
}

_ANCHOR_OFFSETS: dict[StrategyType, range] = {
    StrategyType.LONG_CALL: range(-1, 4),
    StrategyType.LONG_PUT: range(-3, 2),
    StrategyType.CALL_DEBIT_SPREAD: range(-1, 3),
    StrategyType.PUT_DEBIT_SPREAD: range(-2, 2),
    StrategyType.CALL_CREDIT_SPREAD: range(1, 5),
    StrategyType.PUT_CREDIT_SPREAD: range(-4, 0),
    StrategyType.NEUTRAL_DEFINED_RISK: range(2, 5),
}


class StrikeEngine(ABC):
    @abstractmethod
    def rank(
        self, strategy: StrategyType, option_chain: list[OptionQuote], state: MarketState
    ) -> list[StrikeCandidate]: ...


class DeterministicStrikeEngine(StrikeEngine):
    def __init__(self, config: StrikeEngineConfig | None = None) -> None:
        self._config = config or StrikeEngineConfig()

    def rank(
        self, strategy: StrategyType, option_chain: list[OptionQuote], state: MarketState
    ) -> list[StrikeCandidate]:
        cfg = self._config
        if strategy is StrategyType.NO_TRADE:
            return []

        view = structures.ChainView.from_chain(option_chain, state.spot, state)
        if view is None:
            return []

        width_range = self._width_range(state, view)
        # The Strike Engine ranks from MarketState rather than the analysis
        # bundle, so the expected move comes from the state's own volatility
        # slice. None is fine: the breakeven odds are then simply not
        # reported, which is better than reporting them against a guess.
        expected_move = self._expected_move(state)
        scored: list[StrikeCandidate] = []

        for anchor in _ANCHOR_OFFSETS.get(strategy, range(-2, 3)):
            for width in width_range:
                structure = structures.build_structure(
                    strategy,
                    view,
                    anchor_offset=anchor,
                    width_steps=width,
                    max_relative_spread=cfg.max_relative_spread,
                    expected_move=expected_move,
                )
                if structure is None:
                    continue
                rejection = self._reject(structure, view, cfg)
                if rejection is not None:
                    continue
                score, rationale = self._score(structure, state, cfg)
                scored.append(structure.model_copy(update={"score": score, "rationale": rationale}))

        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[: cfg.max_candidates]

    def _expected_move(self, state: MarketState) -> Decimal | None:
        """One sigma to expiry, from spot, ATM IV and calendar time.

        Recomputed here rather than threaded through, because the Strike
        Engine's interface takes a MarketState and not an analysis bundle.
        Same formula as the Volatility brain's, and it returns None on
        anything missing rather than substituting a value — a fabricated
        sigma would put a fabricated probability on every candidate.
        """
        volatility = state.volatility_state
        atm_iv = volatility.atm_iv
        days = volatility.days_to_expiry
        if atm_iv is None or days is None or days <= 0:
            return None
        sigma = float(state.spot) * (atm_iv / 100.0) * math.sqrt(days / 365.0)
        return Decimal(str(round(sigma, 4))) if sigma > 0 else None

    def _width_range(self, state: MarketState, view: structures.ChainView) -> range:
        """Widths worth considering, anchored on the expected move so the
        structure matches how far the market can plausibly travel."""
        analysis = state.analysis
        if analysis is None:
            return range(1, 4)
        expected_move = float(analysis.volatility.expected_move)
        step = float(view.step)
        if expected_move <= 0 or step <= 0:
            return range(1, 4)
        centre = max(1, min(5, round(expected_move / step)))
        return range(max(1, centre - 1), min(7, centre + 2))

    def _reject(
        self,
        candidate: StrikeCandidate,
        view: structures.ChainView,
        cfg: StrikeEngineConfig,
    ) -> str | None:
        """Hard filters. A contract that fails any of these is not proposed at
        all, regardless of how attractive the rest of it looks."""
        if candidate.worst_relative_spread > cfg.max_relative_spread:
            return "spread too wide"

        for leg in candidate.legs:
            quote = view.quote(leg.contract.strike, leg.contract.option_type)
            if quote is None:
                return "missing quote"
            if quote.open_interest < cfg.min_open_interest:
                return "insufficient open interest"
            if quote.bid is None or quote.ask is None:
                return "not two-sided"

        if candidate.max_loss <= 0:
            return "non-positive max loss"
        return None

    def _score(
        self,
        candidate: StrikeCandidate,
        state: MarketState,
        cfg: StrikeEngineConfig,
    ) -> tuple[float, str]:
        delta_fit = self._delta_fit(candidate, cfg)
        structure_fit, structure_note = self._structure_fit(candidate, state, cfg)

        score = ind.blend(
            (delta_fit, cfg.delta_fit_weight),
            (candidate.liquidity_score, cfg.liquidity_weight),
            (structure_fit, cfg.structure_weight),
        )

        strikes = "/".join(
            f"{leg.side.value[0]}{leg.contract.strike:.0f}{leg.contract.option_type.value}"
            for leg in candidate.legs
        )
        premium_kind = "credit" if candidate.is_credit else "debit"
        rationale = (
            f"{strikes}: net {premium_kind} {abs(candidate.net_premium):.0f}, "
            f"max loss {candidate.max_loss:.0f}, "
            f"delta fit {delta_fit:.2f}, liquidity {candidate.liquidity_score:.2f}, "
            f"spread {candidate.worst_relative_spread * 100:.1f}%. {structure_note}"
        )
        return ind.clamp(score or 0.0, 0.0, 1.0), rationale

    def _delta_fit(self, candidate: StrikeCandidate, cfg: StrikeEngineConfig) -> float:
        """How close the primary leg's delta is to the target for this intent.

        Directional structures want meaningful exposure (~0.45); premium
        selling wants a strike that is unlikely to be reached (~0.25).
        """
        target = (
            cfg.credit_target_delta
            if candidate.strategy in _CREDIT_STRATEGIES
            else cfg.directional_target_delta
        )
        primary = self._primary_leg_delta(candidate)
        if primary is None:
            return 0.4
        distance = abs(abs(primary) - target)
        return ind.clamp(1.0 - distance / max(cfg.delta_tolerance, 1e-9), 0.0, 1.0)

    def _primary_leg_delta(self, candidate: StrikeCandidate) -> float | None:
        """For credit structures the short leg defines the risk; for debit
        structures the long leg defines the exposure."""
        wanted = (
            OrderSide.SELL if candidate.strategy in _CREDIT_STRATEGIES else OrderSide.BUY
        )
        for leg in candidate.legs:
            if leg.side is wanted and leg.delta is not None:
                return float(leg.delta)
        for leg in candidate.legs:
            if leg.delta is not None:
                return float(leg.delta)
        return None

    def _structure_fit(
        self,
        candidate: StrikeCandidate,
        state: MarketState,
        cfg: StrikeEngineConfig,
    ) -> tuple[float, str]:
        notes: list[str] = []
        components: list[tuple[float | None, float]] = []

        reward = candidate.net_reward_to_risk
        if reward is not None:
            components.append((ind.clamp(reward / 1.5, 0.0, 1.0), 0.5))
        else:
            # Long premium: reward the placement whose breakeven sits inside
            # the expected move.
            analysis = state.analysis
            expected_move = (
                float(analysis.volatility.expected_move) if analysis is not None else 0.0
            )
            if candidate.breakeven and expected_move > 0:
                distance = abs(float(candidate.breakeven[0]) - float(state.spot))
                components.append((ind.clamp(expected_move / max(distance, 1e-9), 0.0, 1.0), 0.5))
                if distance > expected_move:
                    notes.append("Breakeven sits beyond one sigma")

        wall_penalty, wall_note = self._wall_penalty(candidate, state, cfg)
        if wall_note:
            notes.append(wall_note)
        components.append((1.0 - wall_penalty, 0.3))

        # Capital efficiency: prefer risking less for the same structure.
        if candidate.capital_required > 0:
            efficiency = ind.clamp(
                1.0 - float(candidate.capital_required) / (float(state.spot) * 5.0), 0.0, 1.0
            )
            components.append((efficiency, 0.2))

        score = ind.blend(*components)
        return (score if score is not None else 0.5), " ".join(notes)

    def _wall_penalty(
        self,
        candidate: StrikeCandidate,
        state: MarketState,
        cfg: StrikeEngineConfig,
    ) -> tuple[float, str]:
        """Buying upside straight into a call wall — or downside into a put
        wall — is paying for a move that positioning is leaning against."""
        analysis = state.analysis
        if analysis is None:
            return 0.0, ""

        call_walls = {float(w) for w in analysis.options.call_walls}
        put_walls = {float(w) for w in analysis.options.put_walls}

        for leg in candidate.legs:
            if leg.side is not OrderSide.BUY:
                continue
            strike = float(leg.contract.strike)
            if leg.contract.option_type is OptionType.CE and strike in call_walls:
                return cfg.wall_penalty, f"Long call at the {strike:.0f} call wall."
            if leg.contract.option_type is OptionType.PE and strike in put_walls:
                return cfg.wall_penalty, f"Long put at the {strike:.0f} put wall."

        # Selling *at* a wall is the mirror image: positioning is on your side.
        for leg in candidate.legs:
            if leg.side is not OrderSide.SELL:
                continue
            strike = float(leg.contract.strike)
            if leg.contract.option_type is OptionType.CE and strike in call_walls:
                return 0.0, f"Short call sits at the {strike:.0f} call wall."
            if leg.contract.option_type is OptionType.PE and strike in put_walls:
                return 0.0, f"Short put sits at the {strike:.0f} put wall."
        return 0.0, ""
