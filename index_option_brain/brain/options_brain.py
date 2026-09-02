"""Spec §7. OI must NEVER be treated as a standalone BUY/SELL signal — this
brain reports structure (walls, gamma zones, pressure); direction is decided
only by the Scenario/Signal engines that weigh it against everything else."""

from __future__ import annotations

from abc import ABC, abstractmethod

from index_option_brain.contracts.analysis import OptionsAnalysis
from index_option_brain.contracts.market_state import MarketState


class OptionsBrain(ABC):
    @abstractmethod
    def analyze(self, state: MarketState) -> OptionsAnalysis: ...
