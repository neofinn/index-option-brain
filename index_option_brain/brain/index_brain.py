"""Spec §5. Must NOT choose options, select strikes, size positions, or
execute orders — this brain only characterizes the index itself."""

from __future__ import annotations

from abc import ABC, abstractmethod

from index_option_brain.contracts.analysis import IndexAnalysis
from index_option_brain.contracts.market_state import MarketState


class IndexBrain(ABC):
    @abstractmethod
    def analyze(self, state: MarketState) -> IndexAnalysis: ...
