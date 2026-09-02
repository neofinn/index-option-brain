"""Spec §9. Classifies the environment; must be able to return UNCERTAIN
rather than forcing a classification onto ambiguous conditions."""

from __future__ import annotations

from abc import ABC, abstractmethod

from index_option_brain.contracts.analysis import (
    ConstituentAnalysis,
    IndexAnalysis,
    OptionsAnalysis,
    RegimeState,
    VolatilityAnalysis,
)


class RegimeEngine(ABC):
    @abstractmethod
    def classify(
        self,
        index: IndexAnalysis,
        constituents: ConstituentAnalysis,
        options: OptionsAnalysis,
        volatility: VolatilityAnalysis,
    ) -> RegimeState: ...
