"""Spec §8."""

from __future__ import annotations

from abc import ABC, abstractmethod

from index_option_brain.contracts.analysis import VolatilityAnalysis
from index_option_brain.contracts.market_state import MarketState


class VolatilityEngine(ABC):
    @abstractmethod
    def analyze(self, state: MarketState) -> VolatilityAnalysis: ...
