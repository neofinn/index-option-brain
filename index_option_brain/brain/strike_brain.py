"""Spec §13."""

from __future__ import annotations

from abc import ABC, abstractmethod

from index_option_brain.contracts.enums import StrategyType
from index_option_brain.contracts.instruments import OptionQuote
from index_option_brain.contracts.market_state import MarketState
from index_option_brain.contracts.strike import StrikeCandidate


class StrikeEngine(ABC):
    @abstractmethod
    def rank(
        self, strategy: StrategyType, option_chain: list[OptionQuote], state: MarketState
    ) -> list[StrikeCandidate]: ...
