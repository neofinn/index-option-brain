"""Spec §10. Must NOT immediately convert analysis into BUY/SELL — generates
competing scenarios and must permit NO_TRADE / UNCERTAIN as a legitimate
outcome, not an error case.

Every scenario is scored on its own merits *and* carries the evidence that
argues against it. That matters downstream: the Signal Engine's job is to
check whether one future is genuinely better supported than its rivals, and
it can only do that if the rivals were constructed honestly rather than as
straw men.

A NO_TRADE scenario is always generated, and it scores highest exactly when
it should — when confidence is thin, liquidity is poor, or the directional
cases are indistinguishable.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod

from index_option_brain.brain import indicators as ind
from index_option_brain.brain.config import ScenarioEngineConfig
from index_option_brain.contracts.analysis import AnalysisBundle, RegimeState
from index_option_brain.contracts.enums import (
    BreakoutState,
    Direction,
    IvRegime,
    MarketRegimeType,
    ScenarioKind,
)
from index_option_brain.contracts.market_state import MarketState
from index_option_brain.contracts.scenario import Scenario


class ScenarioEngine(ABC):
    @abstractmethod
    def generate(self, state: MarketState, regime: RegimeState) -> list[Scenario]: ...


class DeterministicScenarioEngine(ScenarioEngine):
    def __init__(self, config: ScenarioEngineConfig | None = None) -> None:
        self._config = config or ScenarioEngineConfig()

    def generate(self, state: MarketState, regime: RegimeState) -> list[Scenario]:
        analysis = state.analysis
        if analysis is None:
            return [
                self._no_trade_scenario(
                    1.0,
                    ["No quantitative analysis attached to this state"],
                )
            ]

        # Each generator returns None when its premise doesn't hold, so a
        # scenario is only argued for when the data actually supports it.
        generated = [
            self._bullish_continuation(state, analysis, regime),
            self._bearish_continuation(state, analysis, regime),
            self._range(state, analysis, regime),
            self._breakout_failure(state, analysis, regime),
            self._reversal(state, analysis, regime),
            self._volatility_scenario(state, analysis, regime),
        ]
        candidates: list[Scenario] = [
            c for c in generated if c is not None and c.score >= self._config.min_score
        ]

        best_directional = max(
            (c.score for c in candidates if c.direction is not Direction.NEUTRAL),
            default=0.0,
        )
        no_trade = self._no_trade_scenario(
            self._no_trade_score(analysis, regime, best_directional),
            self._no_trade_reasons(analysis, regime, best_directional),
        )

        ranked = sorted([*candidates, no_trade], key=lambda s: s.score, reverse=True)
        return ranked[: self._config.max_scenarios]

    def _scenario_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def _bullish_continuation(
        self, state: MarketState, analysis: AnalysisBundle, regime: RegimeState
    ) -> Scenario | None:
        index = analysis.index
        constituents = analysis.constituents
        options = analysis.options
        if index.composite_score <= 0:
            return None

        supporting: list[str] = []
        contradicting: list[str] = []

        base = ind.clamp(index.composite_score, 0.0, 1.0)
        supporting.append(f"Index composite score {index.composite_score:+.2f}")

        breadth_factor = 0.5 + 0.5 * ind.clamp(constituents.breadth_score, 0.0, 1.0)
        if constituents.breadth_score > 0.2:
            supporting.append(f"Breadth confirms ({constituents.advances} advancing)")
        elif constituents.breadth_score < -0.1:
            contradicting.append("Breadth is negative while the index rises")

        if constituents.concentration_score > 0.6:
            contradicting.append(
                f"Move is narrow (concentration {constituents.concentration_score:.2f})"
            )

        if options.oi_structure_score > 0.2:
            supporting.append("Options positioning is supportive below spot")
        elif options.oi_structure_score < -0.2:
            contradicting.append("Call writing above spot caps upside")

        if regime.regime in (MarketRegimeType.TREND_UP, MarketRegimeType.BREAKOUT):
            supporting.append(f"Regime is {regime.regime.value}")
        elif regime.regime in (MarketRegimeType.TREND_DOWN, MarketRegimeType.BREAKDOWN):
            contradicting.append(f"Regime is {regime.regime.value}")

        score = base * breadth_factor * (0.7 + 0.3 * index.confidence)
        score *= 1.0 - 0.15 * len(contradicting)

        confirmation = []
        if index.resistance_levels:
            confirmation.append(f"Acceptance above {index.resistance_levels[0]}")
        confirmation.append("Breadth holding positive into the move")

        invalidation = []
        if index.support_levels:
            invalidation.append(f"Loss of {index.support_levels[0]}")
        if options.call_walls:
            invalidation.append(f"Rejection at the {options.call_walls[0]} call wall")

        return Scenario(
            scenario_id=self._scenario_id(),
            kind=ScenarioKind.BULLISH_CONTINUATION,
            name="Bullish continuation",
            direction=Direction.BULLISH,
            score=ind.clamp(score, 0.0, 1.0),
            confidence=ind.clamp(index.confidence * constituents.confidence, 0.0, 1.0),
            supporting_evidence=supporting,
            contradictory_evidence=contradicting,
            confirmation_conditions=confirmation,
            invalidation_conditions=invalidation,
        )

    def _bearish_continuation(
        self, state: MarketState, analysis: AnalysisBundle, regime: RegimeState
    ) -> Scenario | None:
        index = analysis.index
        constituents = analysis.constituents
        options = analysis.options
        if index.composite_score >= 0:
            return None

        supporting: list[str] = []
        contradicting: list[str] = []

        base = ind.clamp(-index.composite_score, 0.0, 1.0)
        supporting.append(f"Index composite score {index.composite_score:+.2f}")

        breadth_factor = 0.5 + 0.5 * ind.clamp(-constituents.breadth_score, 0.0, 1.0)
        if constituents.breadth_score < -0.2:
            supporting.append(f"Breadth confirms ({constituents.declines} declining)")
        elif constituents.breadth_score > 0.1:
            contradicting.append("Breadth is positive while the index falls")

        if constituents.concentration_score > 0.6:
            contradicting.append(
                f"Decline is narrow (concentration {constituents.concentration_score:.2f})"
            )

        if options.oi_structure_score < -0.2:
            supporting.append("Call writing above spot reinforces the downside")
        elif options.oi_structure_score > 0.2:
            contradicting.append("Put writing below spot is cushioning the fall")

        if regime.regime in (MarketRegimeType.TREND_DOWN, MarketRegimeType.BREAKDOWN):
            supporting.append(f"Regime is {regime.regime.value}")
        elif regime.regime in (MarketRegimeType.TREND_UP, MarketRegimeType.BREAKOUT):
            contradicting.append(f"Regime is {regime.regime.value}")

        score = base * breadth_factor * (0.7 + 0.3 * index.confidence)
        score *= 1.0 - 0.15 * len(contradicting)

        confirmation = []
        if index.support_levels:
            confirmation.append(f"Acceptance below {index.support_levels[0]}")
        confirmation.append("Breadth staying negative into the move")

        invalidation = []
        if index.resistance_levels:
            invalidation.append(f"Reclaim of {index.resistance_levels[0]}")
        if options.put_walls:
            invalidation.append(f"Defence at the {options.put_walls[0]} put wall")

        return Scenario(
            scenario_id=self._scenario_id(),
            kind=ScenarioKind.BEARISH_CONTINUATION,
            name="Bearish continuation",
            direction=Direction.BEARISH,
            score=ind.clamp(score, 0.0, 1.0),
            confidence=ind.clamp(index.confidence * constituents.confidence, 0.0, 1.0),
            supporting_evidence=supporting,
            contradictory_evidence=contradicting,
            confirmation_conditions=confirmation,
            invalidation_conditions=invalidation,
        )

    def _range(
        self, state: MarketState, analysis: AnalysisBundle, regime: RegimeState
    ) -> Scenario | None:
        index = analysis.index
        options = analysis.options

        if index.breakout_state is not BreakoutState.NONE:
            return None

        flatness = ind.clamp(1.0 - abs(index.composite_score) / 0.3, 0.0, 1.0)
        if flatness <= 0:
            return None

        supporting = [f"Index composite is flat at {index.composite_score:+.2f}"]
        contradicting = []
        if options.call_walls and options.put_walls:
            supporting.append(
                f"Chain is bracketed by walls at {options.put_walls[0]} and {options.call_walls[0]}"
            )
        if analysis.volatility.expansion_score > 0.35:
            contradicting.append("Volatility is expanding, which tends to break ranges")

        score = flatness * (0.6 + 0.4 * index.confidence)
        score *= 1.0 - 0.2 * len(contradicting)

        return Scenario(
            scenario_id=self._scenario_id(),
            kind=ScenarioKind.RANGE,
            name="Range / mean reversion",
            direction=Direction.NEUTRAL,
            score=ind.clamp(score, 0.0, 1.0),
            confidence=ind.clamp(index.confidence, 0.0, 1.0),
            supporting_evidence=supporting,
            contradictory_evidence=contradicting,
            confirmation_conditions=["Failure to hold beyond either range extreme"],
            invalidation_conditions=[
                f"Acceptance beyond {index.resistance_levels[0]}"
                if index.resistance_levels
                else "Acceptance beyond the range high",
                f"Acceptance beyond {index.support_levels[0]}"
                if index.support_levels
                else "Acceptance beyond the range low",
            ],
        )

    def _breakout_failure(
        self, state: MarketState, analysis: AnalysisBundle, regime: RegimeState
    ) -> Scenario | None:
        index = analysis.index
        constituents = analysis.constituents

        failed = index.breakout_state in (
            BreakoutState.FAILED_BREAKOUT,
            BreakoutState.FAILED_BREAKDOWN,
        )
        unconfirmed_break = (
            index.breakout_state in (BreakoutState.BREAKOUT, BreakoutState.BREAKDOWN)
            and constituents.participation_score < 0.4
        )
        if not failed and not unconfirmed_break:
            return None

        if failed:
            score = 0.65
            direction = (
                Direction.BEARISH
                if index.breakout_state is BreakoutState.FAILED_BREAKOUT
                else Direction.BULLISH
            )
            supporting = [f"Price rejected its break ({index.breakout_state.value})"]
        else:
            score = 0.45
            direction = (
                Direction.BEARISH
                if index.breakout_state is BreakoutState.BREAKOUT
                else Direction.BULLISH
            )
            supporting = [
                (
                    f"{index.breakout_state.value} is unconfirmed — participation "
                    f"only {constituents.participation_score:.2f}"
                )
            ]

        return Scenario(
            scenario_id=self._scenario_id(),
            kind=ScenarioKind.BREAKOUT_FAILURE,
            name="Breakout failure",
            direction=direction,
            score=ind.clamp(score * (0.6 + 0.4 * index.confidence), 0.0, 1.0),
            confidence=ind.clamp(index.confidence * 0.8, 0.0, 1.0),
            supporting_evidence=supporting,
            contradictory_evidence=(
                ["Breadth still supports the break"]
                if constituents.participation_score > 0.6
                else []
            ),
            confirmation_conditions=["Price closing back inside the prior range"],
            invalidation_conditions=["A second, participated attempt beyond the range"],
        )

    def _reversal(
        self, state: MarketState, analysis: AnalysisBundle, regime: RegimeState
    ) -> Scenario | None:
        index = analysis.index
        divergence = index.trend_score * index.momentum_score
        if divergence >= 0:
            return None

        # Momentum is turning against an established trend — the reversal is
        # in the direction momentum is now pointing.
        direction = Direction.BULLISH if index.momentum_score > 0 else Direction.BEARISH
        score = ind.clamp(abs(divergence) * 1.4, 0.0, 1.0) * (0.6 + 0.4 * index.confidence)

        return Scenario(
            scenario_id=self._scenario_id(),
            kind=ScenarioKind.REVERSAL,
            name="Trend reversal",
            direction=direction,
            score=ind.clamp(score, 0.0, 1.0),
            confidence=ind.clamp(index.confidence * 0.7, 0.0, 1.0),
            supporting_evidence=[
                f"Momentum {index.momentum_score:+.2f} opposes trend {index.trend_score:+.2f}"
            ],
            contradictory_evidence=[
                f"Swing structure still reads {index.structure_score:+.2f}"
            ],
            confirmation_conditions=["Structure break in the direction of momentum"],
            invalidation_conditions=["Trend resuming with a new extreme"],
        )

    def _volatility_scenario(
        self, state: MarketState, analysis: AnalysisBundle, regime: RegimeState
    ) -> Scenario | None:
        volatility = analysis.volatility
        expansion = volatility.expansion_score

        if expansion >= 0.35:
            return Scenario(
                scenario_id=self._scenario_id(),
                kind=ScenarioKind.EXPANSION,
                name="Volatility expansion",
                direction=Direction.NEUTRAL,
                score=ind.clamp(expansion * (0.6 + 0.4 * volatility.confidence), 0.0, 1.0),
                confidence=ind.clamp(volatility.confidence, 0.0, 1.0),
                supporting_evidence=volatility.evidence[:2],
                contradictory_evidence=[],
                confirmation_conditions=["Realized ranges widening with IV"],
                invalidation_conditions=["IV rolling back to its recent mean"],
            )
        if -expansion >= 0.35:
            return Scenario(
                scenario_id=self._scenario_id(),
                kind=ScenarioKind.CONTRACTION,
                name="Volatility contraction",
                direction=Direction.NEUTRAL,
                score=ind.clamp(-expansion * (0.6 + 0.4 * volatility.confidence), 0.0, 1.0),
                confidence=ind.clamp(volatility.confidence, 0.0, 1.0),
                supporting_evidence=volatility.evidence[:2],
                contradictory_evidence=(
                    ["IV is already at the low end of its range"]
                    if volatility.regime is IvRegime.LOW
                    else []
                ),
                confirmation_conditions=["Ranges compressing alongside IV"],
                invalidation_conditions=["An expansion event repricing the chain"],
            )
        return None

    def _no_trade_score(
        self, analysis: AnalysisBundle, regime: RegimeState, best_directional: float
    ) -> float:
        """NO_TRADE gets stronger as the case for anything else gets weaker."""
        thin_conviction = 1.0 - best_directional
        poor_liquidity = 1.0 - analysis.options.liquidity_score
        low_confidence = 1.0 - (
            ind.mean(
                [
                    analysis.index.confidence,
                    analysis.constituents.confidence,
                    analysis.options.confidence,
                ]
            )
            or 0.0
        )
        uncertain_regime = 1.0 if regime.regime is MarketRegimeType.UNCERTAIN else 0.0

        score = ind.blend(
            (thin_conviction, 0.4),
            (poor_liquidity, 0.25),
            (low_confidence, 0.2),
            (uncertain_regime, 0.15),
        )
        return ind.clamp(score or 0.0, 0.0, 1.0)

    def _no_trade_reasons(
        self, analysis: AnalysisBundle, regime: RegimeState, best_directional: float
    ) -> list[str]:
        reasons = []
        if best_directional < 0.4:
            reasons.append(
                f"No directional scenario scores above {best_directional:.2f}"
            )
        if analysis.options.liquidity_score < 0.4:
            reasons.append(
                f"Chain liquidity {analysis.options.liquidity_score:.2f} would surrender any edge"
            )
        if regime.regime is MarketRegimeType.UNCERTAIN:
            reasons.append("Regime is UNCERTAIN")
        if analysis.options.chain_completeness < 0.9:
            reasons.append(
                f"Option chain is only {analysis.options.chain_completeness * 100:.0f}% complete"
            )
        if not reasons:
            reasons.append("Standing aside remains available and is always scored")
        return reasons

    def _no_trade_scenario(self, score: float, reasons: list[str]) -> Scenario:
        return Scenario(
            scenario_id=self._scenario_id(),
            kind=ScenarioKind.NO_TRADE,
            name="No trade",
            direction=Direction.NEUTRAL,
            score=ind.clamp(score, 0.0, 1.0),
            confidence=1.0,
            supporting_evidence=reasons,
            contradictory_evidence=[],
            confirmation_conditions=[],
            invalidation_conditions=[
                "A directional scenario separating clearly from its rivals"
            ],
        )
