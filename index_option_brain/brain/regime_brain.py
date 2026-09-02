"""Spec §9. Classifies the environment; must be able to return UNCERTAIN
rather than forcing a classification onto ambiguous conditions.

Implemented as scored candidates rather than an if/elif cascade, for two
reasons: the runner-up scores are retained as evidence (so a classification
can be argued with), and ambiguity becomes measurable — when the two leading
candidates *contradict each other* and score alike, the honest answer is
UNCERTAIN, not whichever branch happened to be tested first.

Compatible pairs are not treated as ambiguity: TREND_UP and BREAKOUT
describe the same market from two angles, so one winning narrowly over the
other is a classification, not a coin flip.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from index_option_brain.brain import indicators as ind
from index_option_brain.brain.config import RegimeEngineConfig
from index_option_brain.contracts.analysis import (
    ConstituentAnalysis,
    IndexAnalysis,
    OptionsAnalysis,
    RegimeState,
    VolatilityAnalysis,
)
from index_option_brain.contracts.enums import BreakoutState, MarketRegimeType

_BIAS: dict[MarketRegimeType, int] = {
    MarketRegimeType.TREND_UP: 1,
    MarketRegimeType.BREAKOUT: 1,
    MarketRegimeType.TREND_DOWN: -1,
    MarketRegimeType.BREAKDOWN: -1,
    MarketRegimeType.RANGE: 0,
    MarketRegimeType.REVERSAL: 0,
    MarketRegimeType.HIGH_VOLATILITY: 0,
    MarketRegimeType.LOW_VOLATILITY: 0,
    MarketRegimeType.EXPANSION: 0,
    MarketRegimeType.CONTRACTION: 0,
    MarketRegimeType.EXPIRY: 0,
    MarketRegimeType.UNCERTAIN: 0,
}

_DIRECTIONAL = {
    MarketRegimeType.TREND_UP,
    MarketRegimeType.TREND_DOWN,
    MarketRegimeType.BREAKOUT,
    MarketRegimeType.BREAKDOWN,
}


class RegimeEngine(ABC):
    @abstractmethod
    def classify(
        self,
        index: IndexAnalysis,
        constituents: ConstituentAnalysis,
        options: OptionsAnalysis,
        volatility: VolatilityAnalysis,
    ) -> RegimeState: ...


class DeterministicRegimeEngine(RegimeEngine):
    def __init__(self, config: RegimeEngineConfig | None = None) -> None:
        self._config = config or RegimeEngineConfig()

    def classify(
        self,
        index: IndexAnalysis,
        constituents: ConstituentAnalysis,
        options: OptionsAnalysis,
        volatility: VolatilityAnalysis,
    ) -> RegimeState:
        cfg = self._config
        scores = self._score_candidates(index, constituents, options, volatility)
        evidence: list[str] = []
        invalidations: list[str] = []

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best, best_score = ranked[0]
        runner_up, runner_up_score = ranked[1] if len(ranked) > 1 else (best, 0.0)

        # Expiry is a context that overrides structural reads: on expiry day
        # pin risk and collapsing time value dominate whatever the trend says.
        days = volatility.days_to_expiry
        if days is not None and days <= cfg.expiry_days:
            evidence.append(f"{days:.1f} days to expiry — expiry mechanics dominate")
            evidence.append(f"Underlying structural read was {best.value} ({best_score:.2f})")
            return RegimeState(
                regime=MarketRegimeType.EXPIRY,
                confidence=ind.clamp(0.6 + 0.4 * index.confidence, 0.0, 1.0),
                evidence=evidence,
                invalidations=["Expiry context ends with the session"],
                scores={k.value: round(v, 3) for k, v in scores.items()},
            )

        input_confidence = ind.mean(
            [index.confidence, constituents.confidence, options.confidence, volatility.confidence]
        ) or 0.0
        confidence = ind.clamp(best_score * (0.4 + 0.6 * input_confidence), 0.0, 1.0)

        evidence.append(f"Leading classification {best.value} scored {best_score:.2f}")
        evidence.append(f"Runner-up {runner_up.value} scored {runner_up_score:.2f}")
        evidence.extend(index.evidence[:2])

        ambiguous = (
            best_score - runner_up_score
        ) < cfg.separation_threshold and self._contradictory(best, runner_up)

        if best_score < cfg.min_confidence:
            evidence.append(
                f"No classification cleared the {cfg.min_confidence:.2f} floor — reporting UNCERTAIN"
            )
            return RegimeState(
                regime=MarketRegimeType.UNCERTAIN,
                confidence=ind.clamp(confidence, 0.0, 1.0),
                evidence=evidence,
                invalidations=invalidations,
                scores={k.value: round(v, 3) for k, v in scores.items()},
            )

        if ambiguous:
            evidence.append(
                f"{best.value} and {runner_up.value} are contradictory and score within "
                f"{cfg.separation_threshold:.2f} — reporting UNCERTAIN"
            )
            return RegimeState(
                regime=MarketRegimeType.UNCERTAIN,
                confidence=ind.clamp(confidence * 0.6, 0.0, 1.0),
                evidence=evidence,
                invalidations=invalidations,
                scores={k.value: round(v, 3) for k, v in scores.items()},
            )

        if best in _DIRECTIONAL:
            levels = index.support_levels if _BIAS[best] > 0 else index.resistance_levels
            if levels:
                side = "below" if _BIAS[best] > 0 else "above"
                invalidations.append(f"Acceptance {side} {levels[0]} invalidates {best.value}")
        if best is MarketRegimeType.RANGE:
            if index.resistance_levels:
                invalidations.append(f"Break above {index.resistance_levels[0]} ends the range")
            if index.support_levels:
                invalidations.append(f"Break below {index.support_levels[0]} ends the range")

        return RegimeState(
            regime=best,
            confidence=confidence,
            evidence=evidence,
            invalidations=invalidations,
            scores={k.value: round(v, 3) for k, v in scores.items()},
        )

    def _score_candidates(
        self,
        index: IndexAnalysis,
        constituents: ConstituentAnalysis,
        options: OptionsAnalysis,
        volatility: VolatilityAnalysis,
    ) -> dict[MarketRegimeType, float]:
        cfg = self._config
        composite = index.composite_score
        alignment = ind.alignment(
            [index.trend_score, index.structure_score, index.momentum_score]
        )
        participation = constituents.participation_score
        breadth = constituents.breadth_score
        expansion = volatility.expansion_score
        iv_percentile = volatility.iv_percentile
        volatility_level = index.volatility_score

        scores: dict[MarketRegimeType, float] = {}

        # Trend: aligned direction, confirmed by breadth participating.
        trend_strength = ind.clamp(abs(composite) / max(cfg.trend_threshold, 1e-9), 0.0, 1.0)
        trend_quality = trend_strength * alignment * (0.6 + 0.4 * participation)
        if composite > 0:
            scores[MarketRegimeType.TREND_UP] = trend_quality * (
                0.7 + 0.3 * ind.clamp(breadth, 0.0, 1.0)
            )
            scores[MarketRegimeType.TREND_DOWN] = 0.0
        else:
            scores[MarketRegimeType.TREND_DOWN] = trend_quality * (
                0.7 + 0.3 * ind.clamp(-breadth, 0.0, 1.0)
            )
            scores[MarketRegimeType.TREND_UP] = 0.0

        # Breaks: the state machine already confirmed the level; expansion and
        # participation decide whether it deserves to be believed.
        break_quality = 0.55 + 0.25 * ind.clamp(expansion, 0.0, 1.0) + 0.2 * participation
        scores[MarketRegimeType.BREAKOUT] = (
            break_quality if index.breakout_state is BreakoutState.BREAKOUT else 0.0
        )
        scores[MarketRegimeType.BREAKDOWN] = (
            break_quality if index.breakout_state is BreakoutState.BREAKDOWN else 0.0
        )

        # Reversal: a failed break, or momentum fighting the established trend.
        reversal = 0.0
        if index.breakout_state in (
            BreakoutState.FAILED_BREAKOUT,
            BreakoutState.FAILED_BREAKDOWN,
        ):
            reversal = 0.7
        divergence = index.trend_score * index.momentum_score
        if divergence < 0:
            reversal = max(reversal, ind.clamp(abs(divergence) * 1.5, 0.0, 1.0))
        scores[MarketRegimeType.REVERSAL] = reversal

        # Range: nothing is trending, nothing has broken.
        range_score = 0.0
        if index.breakout_state is BreakoutState.NONE:
            flatness = ind.clamp(1.0 - abs(composite) / max(cfg.range_threshold, 1e-9), 0.0, 1.0)
            range_score = flatness * (0.6 + 0.4 * (1.0 - ind.clamp(volatility_level, 0.0, 1.0)))
        scores[MarketRegimeType.RANGE] = range_score

        # Volatility regimes describe a different *axis* from structure: a
        # trend can perfectly well be a volatile trend. Since the contract
        # forces one label, the volatility axis is discounted by how strong
        # the structural read is, so it surfaces when structure has nothing
        # to say rather than displacing a clean, participated trend.
        structural_strength = max(
            scores[MarketRegimeType.TREND_UP],
            scores[MarketRegimeType.TREND_DOWN],
            scores[MarketRegimeType.BREAKOUT],
            scores[MarketRegimeType.BREAKDOWN],
            scores[MarketRegimeType.RANGE],
            reversal,
        )
        axis_discount = 1.0 - ind.clamp(structural_strength, 0.0, 1.0)

        high_volatility = ind.clamp(volatility_level, 0.0, 1.0)
        if iv_percentile is not None:
            high_volatility = max(high_volatility, iv_percentile)
        scores[MarketRegimeType.HIGH_VOLATILITY] = (
            high_volatility * axis_discount
            if high_volatility >= cfg.high_volatility_percentile
            else 0.0
        )
        low_volatility = 1.0 - high_volatility
        scores[MarketRegimeType.LOW_VOLATILITY] = (
            low_volatility * axis_discount
            if high_volatility <= cfg.low_volatility_percentile
            else 0.0
        )

        # Volatility *direction of travel*, which is distinct from its level.
        scores[MarketRegimeType.EXPANSION] = (
            ind.clamp(expansion, 0.0, 1.0) * axis_discount
            if expansion >= cfg.expansion_threshold
            else 0.0
        )
        scores[MarketRegimeType.CONTRACTION] = (
            ind.clamp(-expansion, 0.0, 1.0) * axis_discount
            if -expansion >= cfg.expansion_threshold
            else 0.0
        )

        scores[MarketRegimeType.EXPIRY] = 0.0
        scores[MarketRegimeType.UNCERTAIN] = 0.0
        return scores

    def _contradictory(self, first: MarketRegimeType, second: MarketRegimeType) -> bool:
        """Two classifications conflict when they imply opposite directions,
        or when one says "going somewhere" and the other says "going
        nowhere"."""
        if _BIAS[first] * _BIAS[second] < 0:
            return True
        range_like = {MarketRegimeType.RANGE, MarketRegimeType.CONTRACTION}
        move_like = _DIRECTIONAL | {MarketRegimeType.EXPANSION}
        if (first in range_like and second in move_like) or (
            second in range_like and first in move_like
        ):
            return True
        return MarketRegimeType.REVERSAL in (first, second) and (
            first in _DIRECTIONAL or second in _DIRECTIONAL
        )
