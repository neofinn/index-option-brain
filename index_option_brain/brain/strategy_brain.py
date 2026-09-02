"""Spec §12. NO_TRADE must always be a valid, returnable StrategyCandidate.

The choice of *expression* is separate from the choice of direction, and it
turns mostly on volatility: paying premium when IV is rich is a slow loss
even when the direction is right, and selling premium when IV is cheap
collects too little to pay for the tail. So the same bullish signal produces
a call debit spread in one volatility regime and a put credit spread in
another.

Time to expiry is the other hard constraint: long premium into the last day
or two is a theta trap, so directional views near expiry are only offered as
defined-risk spreads.

Economics on each candidate are indicative — computed from the live chain at
default offsets via `structures.build_structure`, then refined by the Strike
Engine when it picks actual contracts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal

from index_option_brain.brain import indicators as ind
from index_option_brain.brain import structures
from index_option_brain.brain.config import StrategyEngineConfig
from index_option_brain.contracts.analysis import AnalysisBundle
from index_option_brain.contracts.enums import Direction, ScenarioKind, StrategyType
from index_option_brain.contracts.market_state import MarketState
from index_option_brain.contracts.scenario import Scenario
from index_option_brain.contracts.signal import Signal
from index_option_brain.contracts.strategy import StrategyCandidate
from index_option_brain.contracts.strike import StrikeCandidate


class StrategyEngine(ABC):
    @abstractmethod
    def select(self, state: MarketState, signal: Signal) -> list[StrategyCandidate]: ...


class DeterministicStrategyEngine(StrategyEngine):
    def __init__(self, config: StrategyEngineConfig | None = None) -> None:
        self._config = config or StrategyEngineConfig()

    def select(self, state: MarketState, signal: Signal) -> list[StrategyCandidate]:
        cfg = self._config
        analysis = state.analysis
        candidates: list[StrategyCandidate] = []

        if analysis is None:
            return [self._no_trade(["No quantitative analysis attached to this state"])]

        view = structures.ChainView.from_state(state)
        if view is None:
            return [self._no_trade(["Option chain is unusable for structure construction"])]

        blockers = self._blockers(analysis, cfg)

        if signal.direction is Direction.NEUTRAL:
            neutral = self._neutral_candidate(state, analysis, view, blockers, cfg)
            if neutral is not None:
                candidates.append(neutral)
        else:
            candidates.extend(
                self._directional_candidates(signal, analysis, view, blockers, cfg)
            )

        best_score = max((c.score for c in candidates), default=0.0)
        candidates.append(
            self._no_trade(
                blockers or self._no_trade_reasons(best_score, signal),
                score=self._no_trade_score(best_score, cfg),
            )
        )
        return sorted(candidates, key=lambda c: c.score, reverse=True)

    def _no_trade_score(self, best_score: float, cfg: StrategyEngineConfig) -> float:
        """NO_TRADE outranks any structure that fails to clear the acceptance
        floor, so "the only candidate" can never become "the chosen trade"."""
        floor = max(cfg.min_structure_score, 1e-9)
        return ind.clamp(1.0 - best_score / floor, 0.0, 1.0)

    def _blockers(self, analysis: AnalysisBundle, cfg: StrategyEngineConfig) -> list[str]:
        """Conditions under which no structure should be offered at all."""
        blockers: list[str] = []
        if analysis.options.liquidity_score < cfg.min_liquidity_score:
            blockers.append(
                f"Chain liquidity {analysis.options.liquidity_score:.2f} is below the "
                f"{cfg.min_liquidity_score:.2f} minimum"
            )
        if analysis.options.chain_completeness < 0.9:
            blockers.append(
                f"Option chain is only {analysis.options.chain_completeness * 100:.0f}% complete — "
                "spec §29 withholds options entry on incomplete chains"
            )
        return blockers

    def _directional_candidates(
        self,
        signal: Signal,
        analysis: AnalysisBundle,
        view: structures.ChainView,
        blockers: list[str],
        cfg: StrategyEngineConfig,
    ) -> list[StrategyCandidate]:
        if blockers:
            return []

        bullish = signal.direction is Direction.BULLISH
        iv_score = analysis.volatility.iv_score
        days = analysis.volatility.days_to_expiry
        width_steps = self._width_steps(analysis, view, cfg)

        long_premium_allowed = days is None or days >= cfg.min_days_to_expiry_for_long

        plans: list[tuple[StrategyType, int, str]] = []
        if iv_score >= cfg.rich_iv_score:
            reason = f"IV is rich (richness {iv_score:+.2f}) — collect premium rather than pay it"
            plans.append(
                (
                    StrategyType.PUT_CREDIT_SPREAD if bullish else StrategyType.CALL_CREDIT_SPREAD,
                    -2 if bullish else 2,
                    reason,
                )
            )
            plans.append(
                (
                    StrategyType.CALL_DEBIT_SPREAD if bullish else StrategyType.PUT_DEBIT_SPREAD,
                    0,
                    "Defined-risk directional alternative",
                )
            )
        elif iv_score <= cfg.cheap_iv_score:
            reason = f"IV is cheap (richness {iv_score:+.2f}) — paying premium is efficient here"
            if long_premium_allowed:
                plans.append(
                    (StrategyType.LONG_CALL if bullish else StrategyType.LONG_PUT, 0, reason)
                )
            plans.append(
                (
                    StrategyType.CALL_DEBIT_SPREAD if bullish else StrategyType.PUT_DEBIT_SPREAD,
                    0,
                    "Caps cost while keeping the directional exposure",
                )
            )
        else:
            plans.append(
                (
                    StrategyType.CALL_DEBIT_SPREAD if bullish else StrategyType.PUT_DEBIT_SPREAD,
                    0,
                    (
                        f"IV is neither rich nor cheap (richness {iv_score:+.2f}) — "
                        "a defined-risk spread avoids taking a volatility view"
                    ),
                )
            )
            if long_premium_allowed and signal.score > 0.6:
                plans.append(
                    (
                        StrategyType.LONG_CALL if bullish else StrategyType.LONG_PUT,
                        0,
                        "High conviction justifies uncapped upside",
                    )
                )

        if not long_premium_allowed:
            note = (
                f"{days:.1f} days to expiry — long premium excluded as a theta trap"
                if days is not None
                else ""
            )
        else:
            note = ""

        out: list[StrategyCandidate] = []
        for strategy, anchor, reason in plans:
            structure = structures.build_structure(
                strategy,
                view,
                anchor_offset=anchor,
                width_steps=width_steps,
                expected_move=analysis.volatility.expected_move,
            )
            if structure is None:
                continue
            score, rationale = self._score(structure, signal, analysis, cfg)
            if note:
                rationale = f"{rationale} {note}"
            out.append(self._to_candidate(structure, score, f"{reason}. {rationale}"))
        return out

    def _neutral_candidate(
        self,
        state: MarketState,
        analysis: AnalysisBundle,
        view: structures.ChainView,
        blockers: list[str],
        cfg: StrategyEngineConfig,
    ) -> StrategyCandidate | None:
        """A neutral signal is the absence of a *directional* opportunity, not
        necessarily the absence of any. A well-supported range with rich
        premium is exactly what defined-risk neutral structures exist for."""
        if blockers:
            return None

        leader = self._leading_scenario(state.active_scenarios)
        if leader is None or leader.kind not in (ScenarioKind.RANGE, ScenarioKind.CONTRACTION):
            return None
        if leader.score < 0.45:
            return None
        if analysis.volatility.iv_score < cfg.rich_iv_score:
            return None

        # Short strikes sit around one sigma out; the protective wings are
        # only a few steps beyond them. Sizing the wings to a second full
        # sigma would demand a chain far wider than one that exists and would
        # collateralize far more than the structure risks.
        short_offset = max(2, self._width_steps(analysis, view, cfg))
        wing_width = max(1, min(3, short_offset // 2))
        structure = structures.build_structure(
            StrategyType.NEUTRAL_DEFINED_RISK,
            view,
            anchor_offset=short_offset,
            width_steps=wing_width,
            expected_move=analysis.volatility.expected_move,
        )
        if structure is None:
            return None

        score = ind.clamp(
            leader.score * (0.6 + 0.4 * analysis.options.liquidity_score), 0.0, 1.0
        )
        rationale = (
            f"{leader.name} scored {leader.score:.2f} with rich premium "
            f"(richness {analysis.volatility.iv_score:+.2f}); defined-risk neutral structure "
            f"collects {abs(structure.net_premium):.0f} against {structure.max_loss:.0f} max loss"
        )
        return self._to_candidate(structure, score, rationale)

    def _leading_scenario(self, scenarios: list[Scenario]) -> Scenario | None:
        tradeable = [s for s in scenarios if s.kind is not ScenarioKind.NO_TRADE]
        if not tradeable:
            return None
        return max(tradeable, key=lambda s: s.score)

    def _width_steps(
        self,
        analysis: AnalysisBundle,
        view: structures.ChainView,
        cfg: StrategyEngineConfig,
    ) -> int:
        """Spread width scaled to the expected move, so the structure is sized
        to how far the market can plausibly travel before expiry rather than
        to an arbitrary strike count."""
        expected_move = float(analysis.volatility.expected_move)
        step = float(view.step)
        if expected_move <= 0 or step <= 0:
            return 2
        steps = round((expected_move * cfg.spread_width_expected_move) / step)
        return max(1, min(10, steps))

    def _score(
        self,
        structure: StrikeCandidate,
        signal: Signal,
        analysis: AnalysisBundle,
        cfg: StrategyEngineConfig,
    ) -> tuple[float, str]:
        iv_score = analysis.volatility.iv_score
        is_credit = structure.is_credit

        # Does the structure agree with the volatility view it implies?
        volatility_fit = ind.clamp(0.5 + (iv_score if is_credit else -iv_score), 0.0, 1.0)

        days = analysis.volatility.days_to_expiry
        if days is None:
            theta_fit = 0.6
        elif is_credit:
            theta_fit = ind.clamp(1.0 - days / 30.0, 0.3, 1.0)
        else:
            theta_fit = ind.clamp(days / 10.0, 0.2, 1.0)

        # Net of costs, deliberately. Gross reward-to-risk favours far-OTM
        # spreads precisely where flat-fee brokerage takes the largest
        # share of the credit.
        reward = structure.net_reward_to_risk
        if reward is None:
            # Long premium: judge by how far the breakeven sits inside one
            # sigma, since an unreachable breakeven is not "unlimited upside".
            expected_move = float(analysis.volatility.expected_move)
            breakeven_distance = (
                abs(float(structure.breakeven[0]) - float(analysis.options.atm_strike or 0))
                if structure.breakeven and analysis.options.atm_strike
                else 0.0
            )
            reward_fit = (
                ind.clamp(expected_move / breakeven_distance, 0.0, 1.0)
                if breakeven_distance > 0 and expected_move > 0
                else 0.4
            )
        else:
            reward_fit = ind.clamp(reward / max(cfg.min_reward_to_risk, 1e-9), 0.0, 1.0)

        # The fit components describe how well this structure would express a
        # view *given that there is one*. So conviction is a ceiling rather
        # than one vote among several: without it, excellent volatility and
        # liquidity conditions would score a trade that has no directional
        # edge behind it at all.
        quality = ind.blend(
            (volatility_fit, 0.35),
            (theta_fit, 0.20),
            (reward_fit, 0.25),
            (structure.liquidity_score, 0.20),
        )
        quality = quality if quality is not None else 0.5
        score = signal.score * (0.55 + 0.45 * quality)

        rationale = (
            f"signal {signal.score:.2f} x quality {quality:.2f} "
            f"(volatility fit {volatility_fit:.2f}, theta fit {theta_fit:.2f}, "
            f"reward fit {reward_fit:.2f}, liquidity {structure.liquidity_score:.2f})."
        )
        return ind.clamp(score, 0.0, 1.0), rationale

    def _to_candidate(
        self, structure: StrikeCandidate, score: float, rationale: str
    ) -> StrategyCandidate:
        return StrategyCandidate(
            strategy=structure.strategy,
            score=score,
            max_loss=structure.max_loss,
            max_profit=structure.max_profit,
            breakeven=list(structure.breakeven),
            rationale=rationale,
        )

    def _no_trade_reasons(self, best_score: float, signal: Signal) -> list[str]:
        if signal.direction is Direction.NEUTRAL:
            reasons = ["No directional signal, and no neutral structure is warranted"]
        else:
            reasons = [f"Best available structure scored {best_score:.2f}"]
        reasons.extend(signal.contradictions[:3])
        return reasons

    def _no_trade(self, reasons: list[str], score: float = 1.0) -> StrategyCandidate:
        return StrategyCandidate(
            strategy=StrategyType.NO_TRADE,
            score=score,
            max_loss=Decimal(0),
            max_profit=Decimal(0),
            breakeven=[],
            rationale="; ".join(reasons) if reasons else "Standing aside is always available",
        )
