"""Turn the analysis into a paragraph, without a model.

Most of what people want an LLM for here is explanation, and explanation
mostly does not need one. The brains already emit structured evidence with
every score — the missing piece is ordering it and writing it down, which is
deterministic work.

Doing it without a model is worth the effort for three reasons. It is
available when `LLM_ENABLED=false`, which is the supported configuration. It
costs nothing per call, so the console can render it on every cycle. And it
cannot hallucinate: every sentence below is assembled from a field that was
measured, so there is no failure mode where the summary describes a market
the engine did not observe.

What is genuinely left for a model
----------------------------------
The jobs that need knowledge from outside the feed:

* explaining *why* an anomaly happened — the straddle diverging from IV says
  something is wrong and cannot say what;
* the calendar the four unreachable triggers need;
* a post-mortem on a thesis that failed, in terms the numbers do not carry.

Those are the cases where an `AIProvider` earns its place. This is not one of
them, and building it as though it were would have made the console depend on
an API key to explain itself.
"""

from __future__ import annotations

from index_option_brain.agent.intelligence_provider import (
    AgentAssessment,
    IntelligenceProvider,
)
from index_option_brain.contracts.analysis import AnalysisBundle, RegimeState
from index_option_brain.contracts.enums import Direction, MarketRegimeType, StrategyType
from index_option_brain.contracts.signal import Signal
from index_option_brain.contracts.strike import StrikeCandidate

# How many evidence lines to carry from each domain. Enough to justify the
# read, few enough that the summary stays readable — an explanation nobody
# finishes is not an explanation.
_PER_DOMAIN = 2


class NarrativeProvider(IntelligenceProvider):
    """Composes an assessment from measured fields only.

    Not an `AnalysisContext` consumer in the general sense: it takes the
    contracts directly, because it is the one provider that needs no tools —
    everything it says is already in the analysis.
    """

    async def analyze(self, context: object) -> AgentAssessment:
        """Present for interface compatibility.

        The useful entry point is `describe`, which takes the analysis
        directly. This exists so a `NarrativeProvider` can stand in wherever
        an `IntelligenceProvider` is expected without a special case.
        """
        return AgentAssessment(
            summary="NarrativeProvider needs the analysis; call describe().",
            provider="narrative",
        )

    def describe(
        self,
        *,
        analysis: AnalysisBundle | None,
        regime: RegimeState | None,
        signal: Signal | None,
        strategy: StrategyType,
        candidate: StrikeCandidate | None = None,
        is_authorized: bool = False,
    ) -> AgentAssessment:
        """The cycle, in prose."""
        if analysis is None:
            return AgentAssessment(
                summary=(
                    "No analysis was produced, so there is nothing to explain. "
                    "That is a data problem rather than a market read."
                ),
                unknowns=["The pipeline did not reach the analysis stage"],
                provider="narrative",
            )

        supporting: list[str] = []
        contradicting: list[str] = []
        unknowns: list[str] = []
        sources: list[str] = []

        for name, part in (
            ("index", analysis.index),
            ("constituents", analysis.constituents),
            ("options", analysis.options),
            ("volatility", analysis.volatility),
        ):
            # Confidence at zero means the domain measured nothing. Reporting
            # its evidence anyway would present an absence as a finding, which
            # is the specific failure the Regime Engine's coverage gate exists
            # to prevent.
            if part.confidence <= 0.0:
                unknowns.append(f"The {name} brain measured nothing this cycle")
                continue
            lines = list(part.evidence)[:_PER_DOMAIN]
            if lines:
                supporting.extend(f"{name}: {line}" for line in lines)
                sources.append(f"{name} analysis (confidence {part.confidence:.2f})")

        if regime is not None:
            if regime.regime is MarketRegimeType.UNCERTAIN:
                contradicting.extend(regime.evidence[:_PER_DOMAIN])
            else:
                supporting.extend(f"regime: {line}" for line in regime.evidence[:1])
            sources.append("regime engine")

        if signal is not None and signal.evidence:
            bucket = (
                contradicting if signal.direction is Direction.NEUTRAL else supporting
            )
            bucket.extend(f"signal: {line}" for line in signal.evidence[:_PER_DOMAIN])
            sources.append("signal engine")

        return AgentAssessment(
            summary=self._summary(
                regime, signal, strategy, candidate, is_authorized, analysis
            ),
            supporting_points=supporting,
            contradicting_points=contradicting,
            unknowns=unknowns,
            sources=sources,
            provider="narrative",
        )

    def _summary(
        self,
        regime: RegimeState | None,
        signal: Signal | None,
        strategy: StrategyType,
        candidate: StrikeCandidate | None,
        is_authorized: bool,
        analysis: AnalysisBundle,
    ) -> str:
        parts: list[str] = []

        if regime is None:
            parts.append("No regime was classified")
        elif regime.regime is MarketRegimeType.UNCERTAIN:
            parts.append(
                f"The regime is UNCERTAIN (confidence {regime.confidence:.2f})"
            )
        else:
            parts.append(
                f"The market reads as {regime.regime.value.replace('_', ' ').lower()} "
                f"with {regime.confidence:.0%} confidence"
            )

        if signal is not None:
            if signal.direction is Direction.NEUTRAL:
                parts.append("no directional signal cleared the gates")
            else:
                parts.append(
                    f"the signal is {signal.direction.value} at {signal.score:.2f}"
                )

        volatility = analysis.volatility
        if volatility.iv_percentile is not None:
            stance = (
                "favours paying premium"
                if volatility.iv_score < 0
                else "favours collecting premium"
            )
            parts.append(
                f"implied volatility sits at the {volatility.iv_percentile:.0%} "
                f"percentile, which {stance}"
            )
        if volatility.expected_move > 0:
            parts.append(
                f"the market is pricing a {float(volatility.expected_move):.0f}-point "
                "one-sigma move to expiry"
            )

        if strategy is StrategyType.NO_TRADE:
            parts.append("so the selected action is to stand aside")
        else:
            action = strategy.value.replace("_", " ").lower()
            if candidate is not None and candidate.breakeven_sigmas is not None:
                parts.append(
                    f"so a {action} is preferred, needing a "
                    f"{candidate.breakeven_sigmas:.2f}-sigma move to break even"
                )
            else:
                parts.append(f"so a {action} is preferred")
            parts.append(
                "risk has authorized a size"
                if is_authorized
                else "nothing has been authorized"
            )

        return ". ".join(part[0].upper() + part[1:] for part in parts if part) + "."
