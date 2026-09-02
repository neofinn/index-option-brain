"""Spec §6. Must distinguish broad participation from index movement driven
by a handful of heavyweight constituents.

That distinction is the whole point of this brain: a +0.6% index printed by
four heavyweights while 34 names fall is a materially different market from
a +0.6% index with 40 names up, and the options structure that suits each is
different too. `breadth_score` measures how many names participate,
`concentration_score` measures how few names are responsible, and
`participation_score` measures whether those two agree.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from index_option_brain.brain import indicators as ind
from index_option_brain.brain.config import ConstituentBrainConfig
from index_option_brain.contracts.analysis import ConstituentAnalysis
from index_option_brain.contracts.market_state import MarketState


class ConstituentBrain(ABC):
    @abstractmethod
    def analyze(self, state: MarketState) -> ConstituentAnalysis: ...


class DeterministicConstituentBrain(ConstituentBrain):
    def __init__(self, config: ConstituentBrainConfig | None = None) -> None:
        self._config = config or ConstituentBrainConfig()

    def analyze(self, state: MarketState) -> ConstituentAnalysis:
        cfg = self._config
        constituent_state = state.constituent_state
        quotes = constituent_state.quotes
        weights = constituent_state.weights
        sectors = constituent_state.sectors

        if not quotes:
            return ConstituentAnalysis(
                breadth_score=0.0,
                participation_score=0.0,
                leadership_score=0.0,
                concentration_score=0.0,
                confidence=0.0,
                evidence=["No constituent quotes available"],
            )

        evidence: list[str] = []
        changes: dict[str, float] = {}
        contributions: dict[str, float] = {}

        for quote in quotes:
            change_pct = float(quote.change_pct)
            changes[quote.symbol] = change_pct
            weight = weights.get(quote.symbol)
            if weight is not None:
                # Index points move roughly with weight x the name's move.
                contributions[quote.symbol] = weight * change_pct / 100.0

        advances = sum(1 for c in changes.values() if c > cfg.unchanged_threshold_pct)
        declines = sum(1 for c in changes.values() if c < -cfg.unchanged_threshold_pct)
        unchanged = len(changes) - advances - declines

        breadth_score = 0.0
        if advances + declines > 0:
            raw_breadth = (advances - declines) / (advances + declines)
            breadth_score = ind.squash(raw_breadth, cfg.breadth_scale)
            evidence.append(f"Breadth {advances} advancing / {declines} declining")

        weighted_change_pct = sum(contributions.values())
        contribution_score = ind.squash(weighted_change_pct, cfg.contribution_scale)
        if contributions:
            evidence.append(
                f"Weight-adjusted move of observed constituents: {weighted_change_pct:+.2f}%"
            )

        concentration_score = ind.normalized_hhi(list(contributions.values())) or 0.0

        leadership_score, leadership_evidence = self._leadership(contributions, cfg)
        evidence.extend(leadership_evidence)

        participation_score, participation_evidence = self._participation(
            changes,
            float(state.index_state.quote.change_pct),
            weighted_change_pct,
            concentration_score,
        )
        evidence.extend(participation_evidence)

        ranked = sorted(contributions.items(), key=lambda item: item[1], reverse=True)
        top_contributors = [symbol for symbol, value in ranked if value > 0][: cfg.top_names]
        top_detractors = [symbol for symbol, value in reversed(ranked) if value < 0][
            : cfg.top_names
        ]

        sector_scores = self._sector_scores(contributions, sectors, cfg)
        if sector_scores:
            leader = max(sector_scores.items(), key=lambda item: item[1])
            laggard = min(sector_scores.items(), key=lambda item: item[1])
            evidence.append(
                f"Sector leadership: {leader[0]} {leader[1]:+.2f}, "
                f"laggard: {laggard[0]} {laggard[1]:+.2f}"
            )

        observed_weight = sum(weights.get(q.symbol, 0.0) for q in quotes)
        total_weight = sum(weights.values()) if weights else 0.0
        coverage = observed_weight / total_weight if total_weight > 0 else 0.0

        confidence = self._confidence(coverage, len(quotes), breadth_score, contribution_score, cfg)

        return ConstituentAnalysis(
            breadth_score=ind.clamp(breadth_score),
            participation_score=ind.clamp(participation_score, 0.0, 1.0),
            leadership_score=ind.clamp(leadership_score),
            concentration_score=ind.clamp(concentration_score, 0.0, 1.0),
            sector_scores=sector_scores,
            top_contributors=top_contributors,
            top_detractors=top_detractors,
            confidence=confidence,
            evidence=evidence,
            advances=advances,
            declines=declines,
            unchanged=unchanged,
            weighted_change_pct=weighted_change_pct,
            weight_coverage=coverage,
            contributions=contributions,
        )

    def _leadership(
        self, contributions: dict[str, float], cfg: ConstituentBrainConfig
    ) -> tuple[float, list[str]]:
        """Signed share of total movement owned by the largest movers."""
        if not contributions:
            return 0.0, []
        ranked = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)
        leaders = ranked[: cfg.leaders]
        total_magnitude = sum(abs(v) for v in contributions.values())
        if total_magnitude <= 0:
            return 0.0, []
        net_leader_contribution = sum(v for _, v in leaders)
        score = net_leader_contribution / total_magnitude
        names = ", ".join(symbol for symbol, _ in leaders)
        share = sum(abs(v) for _, v in leaders) / total_magnitude
        return score, [f"Top {len(leaders)} movers ({names}) account for {share * 100:.0f}% of the move"]

    def _participation(
        self,
        changes: dict[str, float],
        index_change_pct: float,
        weighted_change_pct: float,
        concentration_score: float,
    ) -> tuple[float, list[str]]:
        """0..1 quality measure: are the constituents actually confirming the
        *index's* move, or is a narrow group carrying it?

        The reference direction is the index's own change, not the sum of
        observed contributions. Measuring against the contribution sum would
        be circular — a rally carried by two heavyweights while thirty names
        fall would score high participation, since the observed names would
        agree with their own aggregate.
        """
        if not changes:
            return 0.0, []

        reference = index_change_pct if index_change_pct != 0 else weighted_change_pct
        if reference == 0:
            return 0.0, ["Index is unchanged — participation is undefined"]

        index_direction = 1.0 if reference > 0 else -1.0
        agreeing = sum(1 for change in changes.values() if change * index_direction > 0)
        agreement_ratio = agreeing / len(changes)

        # Broad agreement is good; heavy concentration discounts it.
        score = agreement_ratio * (1.0 - 0.5 * concentration_score)
        evidence = [
            (
                f"{agreeing}/{len(changes)} constituents moving with the index "
                f"({reference:+.2f}%, concentration {concentration_score:.2f})"
            )
        ]
        if agreement_ratio < 0.5:
            evidence.append(
                "Most constituents are moving against the index — the move is not broad-based"
            )
        if concentration_score > 0.6:
            evidence.append(
                "Move is narrow — driven by a few heavyweights, not broad participation"
            )
        return score, evidence

    def _sector_scores(
        self,
        contributions: dict[str, float],
        sectors: dict[str, str],
        cfg: ConstituentBrainConfig,
    ) -> dict[str, float]:
        if not sectors:
            return {}
        totals: dict[str, float] = {}
        for symbol, contribution in contributions.items():
            sector = sectors.get(symbol)
            if sector is None:
                continue
            totals[sector] = totals.get(sector, 0.0) + contribution
        return {
            sector: ind.clamp(ind.squash(value, cfg.contribution_scale))
            for sector, value in totals.items()
        }

    def _confidence(
        self,
        coverage: float,
        observed: int,
        breadth_score: float,
        contribution_score: float,
        cfg: ConstituentBrainConfig,
    ) -> float:
        if observed == 0:
            return 0.0
        coverage_factor = ind.clamp(coverage / cfg.min_coverage, 0.0, 1.0)
        agreement = ind.alignment([breadth_score, contribution_score])
        strength = (abs(breadth_score) + abs(contribution_score)) / 2
        return ind.clamp(coverage_factor * (0.4 + 0.6 * agreement) * (0.4 + 0.6 * strength), 0.0, 1.0)
