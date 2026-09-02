"""Spec §12. NO_TRADE must always be a valid, returnable StrategyCandidate."""

from __future__ import annotations

from abc import ABC, abstractmethod

from index_option_brain.contracts.market_state import MarketState
from index_option_brain.contracts.signal import Signal
from index_option_brain.contracts.strategy import StrategyCandidate


class StrategyEngine(ABC):
    @abstractmethod
    def select(self, state: MarketState, signal: Signal) -> list[StrategyCandidate]: ...
