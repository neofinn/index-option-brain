"""Spec §11. A Signal is NOT an order, and must not come from primitive
single-indicator logic.

Four independent conditions must all hold before this engine will emit a
directional signal, and each one is a separate way for a naive setup to be
rejected:

  1. **Scenario separation** — the leading future must out-score the best
     future pointing the *other* way by a configured margin. Two futures
     that score alike mean the evidence does not distinguish them.
  2. **Cross-domain agreement** — index, breadth, and options positioning
     must broadly agree. A single strong reading dragging two contradictions
     behind it fails `alignment`.
  3. **Primary-domain participation** — the index domain must actually vote
     for the direction, and at least two domains must express a view at all.
     Agreement is not enough on its own: a domain that abstains neither
     agrees nor disagrees, so one lone non-zero domain would otherwise score
     as unanimous. This gate is what structurally prevents options
     positioning from carrying a trade by itself (spec §7).
  4. **Absolute conviction** — the combined score must clear a floor.

Anything short of that is still returned, as a NEUTRAL signal carrying the
contradictions that stopped it — which is what lets the Strategy Engine
answer NO_TRADE with reasons rather than silence.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod

from index_option_brain.brain import indicators as ind
from index_option_brain.brain.config import SignalEngineConfig
from index_option_brain.contracts.analysis import AnalysisBundle
from index_option_brain.contracts.enums import Direction, MarketRegimeType, ScenarioKind
from index_option_brain.contracts.market_state import MarketState
from index_option_brain.contracts.scenario import Scenario
from index_option_brain.contracts.signal import Signal


class SignalEngine(ABC):
    @abstractmethod
    def evaluate(self, state: MarketState, scenarios: list[Scenario]) -> Signal: ...


class DeterministicSignalEngine(SignalEngine):
    def __init__(self, config: SignalEngineConfig | None = None) -> None:
        self._config = config or SignalEngineConfig()

    def evaluate(self, state: MarketState, scenarios: list[Scenario]) -> Signal:
        cfg = self._config
        analysis = state.analysis

        if not scenarios:
            return self._neutral_signal("", ["No scenarios were generated"])
        if analysis is None:
            return self._neutral_signal(
                scenarios[0].scenario_id, ["No quantitative analysis attached to this state"]
            )

        leader = max(scenarios, key=lambda s: s.score)
        evidence: list[str] = [f"Leading scenario: {leader.name} ({leader.score:.2f})"]
        contradictions: list[str] = list(leader.contradictory_evidence)

        if leader.kind is ScenarioKind.NO_TRADE:
            evidence.extend(leader.supporting_evidence)
            return self._neutral_signal(leader.scenario_id, contradictions, evidence)

        if leader.direction is Direction.NEUTRAL:
            evidence.append(
                f"{leader.name} does not imply a direction — no directional signal"
            )
            return self._neutral_signal(leader.scenario_id, contradictions, evidence)

        separation, separation_note = self._separation(leader, scenarios)
        evidence.append(separation_note)

        domain_scores, domain_evidence, domain_contradictions = self._domains(
            analysis, leader.direction
        )
        evidence.extend(domain_evidence)
        contradictions.extend(domain_contradictions)

        index_vote = domain_scores[0] or 0.0
        participating = sum(1 for vote in domain_scores if vote is not None and abs(vote) > 0.05)

        agreement = ind.alignment(domain_scores)
        evidence.append(
            f"Cross-domain agreement {agreement:.2f} across {participating} "
            f"participating of {len(domain_scores)} domains"
        )

        regime_factor, regime_note = self._regime_factor(state, leader.direction)
        if regime_note:
            if regime_factor < 1.0:
                contradictions.append(regime_note)
            else:
                evidence.append(regime_note)

        separation_factor = ind.clamp(separation / max(cfg.min_separation, 1e-9), 0.0, 1.0)
        score = leader.score * agreement * regime_factor * (0.5 + 0.5 * separation_factor)

        failed_gates: list[str] = []
        if separation < cfg.min_separation:
            failed_gates.append(
                f"Scenario separation {separation:.2f} below the {cfg.min_separation:.2f} minimum"
            )
        if agreement < cfg.min_alignment:
            failed_gates.append(
                f"Domain agreement {agreement:.2f} below the {cfg.min_alignment:.2f} minimum"
            )
        if index_vote < cfg.min_primary_vote:
            failed_gates.append(
                f"Index domain votes only {index_vote:+.2f} for this direction — "
                "options positioning and breadth cannot authorize it alone"
            )
        if participating < cfg.min_participating_domains:
            failed_gates.append(
                f"Only {participating} domain(s) express a view; "
                f"{cfg.min_participating_domains} required for conviction"
            )
        if score < cfg.min_score:
            failed_gates.append(
                f"Combined score {score:.2f} below the {cfg.min_score:.2f} minimum"
            )

        if failed_gates:
            contradictions.extend(failed_gates)
            return self._neutral_signal(
                leader.scenario_id,
                contradictions,
                [*evidence, "Directional signal withheld — gates not satisfied"],
                score=score,
            )

        return Signal(
            signal_id=uuid.uuid4().hex[:12],
            direction=leader.direction,
            score=ind.clamp(score, 0.0, 1.0),
            confidence=ind.clamp(leader.confidence * agreement, 0.0, 1.0),
            selected_scenario_id=leader.scenario_id,
            evidence=evidence,
            contradictions=contradictions,
            confirmation_required=score < cfg.confirmation_score or bool(contradictions),
            invalidation_conditions=leader.invalidation_conditions,
        )

    def _separation(
        self, leader: Scenario, scenarios: list[Scenario]
    ) -> tuple[float, str]:
        """Distance from the best *opposing* future, not merely the runner-up.

        A second bullish scenario scoring alongside the leading bullish one is
        corroboration; a bearish one scoring alongside it is a coin flip.
        """
        opposing = [
            s
            for s in scenarios
            if s.scenario_id != leader.scenario_id
            and s.direction is not leader.direction
        ]
        if not opposing:
            return 1.0, "No opposing scenario was generated"
        best_opposing = max(opposing, key=lambda s: s.score)
        separation = leader.score - best_opposing.score
        note = (
            f"Separation {separation:+.2f} over the best opposing case "
            f"({best_opposing.name} {best_opposing.score:.2f})"
        )
        return separation, note

    def _domains(
        self, analysis: AnalysisBundle, direction: Direction
    ) -> tuple[list[float | None], list[str], list[str]]:
        """Signed votes from each domain, oriented so positive means "agrees
        with the proposed direction"."""
        sign = 1.0 if direction is Direction.BULLISH else -1.0
        evidence: list[str] = []
        contradictions: list[str] = []

        index_vote = analysis.index.composite_score * sign
        evidence.append(f"Index domain votes {index_vote:+.2f}")
        if index_vote < 0:
            contradictions.append("Index structure opposes the proposed direction")

        breadth_vote = analysis.constituents.breadth_score * sign
        breadth_vote *= 0.5 + 0.5 * analysis.constituents.participation_score
        evidence.append(f"Breadth domain votes {breadth_vote:+.2f}")
        if breadth_vote < 0:
            contradictions.append("Breadth opposes the proposed direction")

        # Options positioning is a corroborating domain only. Note it is
        # weighted below the others: OI structure is never permitted to be
        # the deciding vote (spec §7).
        options_vote = analysis.options.oi_structure_score * sign * 0.7
        evidence.append(f"Options positioning votes {options_vote:+.2f}")
        if options_vote < -0.2:
            contradictions.append("Options positioning opposes the proposed direction")

        if analysis.options.liquidity_score < 0.35:
            contradictions.append(
                f"Chain liquidity {analysis.options.liquidity_score:.2f} is too thin to express this"
            )

        return [index_vote, breadth_vote, options_vote], evidence, contradictions

    def _regime_factor(self, state: MarketState, direction: Direction) -> tuple[float, str]:
        regime = state.market_regime
        if regime is None:
            return 1.0, ""

        supportive = {
            Direction.BULLISH: {MarketRegimeType.TREND_UP, MarketRegimeType.BREAKOUT},
            Direction.BEARISH: {MarketRegimeType.TREND_DOWN, MarketRegimeType.BREAKDOWN},
        }.get(direction, set())
        opposing = {
            Direction.BULLISH: {MarketRegimeType.TREND_DOWN, MarketRegimeType.BREAKDOWN},
            Direction.BEARISH: {MarketRegimeType.TREND_UP, MarketRegimeType.BREAKOUT},
        }.get(direction, set())

        if regime.regime in supportive:
            return 1.0, f"Regime {regime.regime.value} supports the direction"
        if regime.regime in opposing:
            return 0.5, f"Regime {regime.regime.value} opposes the direction"
        if regime.regime is MarketRegimeType.UNCERTAIN:
            return 0.7, "Regime is UNCERTAIN, which discounts conviction"
        if regime.regime is MarketRegimeType.EXPIRY:
            return 0.75, "Expiry mechanics discount directional conviction"
        return 0.85, f"Regime {regime.regime.value} is neutral for this direction"

    def _neutral_signal(
        self,
        scenario_id: str,
        contradictions: list[str],
        evidence: list[str] | None = None,
        score: float = 0.0,
    ) -> Signal:
        return Signal(
            signal_id=uuid.uuid4().hex[:12],
            direction=Direction.NEUTRAL,
            score=ind.clamp(score, 0.0, 1.0),
            confidence=0.0,
            selected_scenario_id=scenario_id,
            evidence=evidence or [],
            contradictions=contradictions,
            confirmation_required=True,
            invalidation_conditions=[],
        )
