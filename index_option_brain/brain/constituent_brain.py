"""Spec §6. Must distinguish broad participation from index movement driven
by a handful of heavyweight constituents."""

from __future__ import annotations

from abc import ABC, abstractmethod

from index_option_brain.contracts.analysis import ConstituentAnalysis
from index_option_brain.contracts.market_state import MarketState


class ConstituentBrain(ABC):
    @abstractmethod
    def analyze(self, state: MarketState) -> ConstituentAnalysis: ...
