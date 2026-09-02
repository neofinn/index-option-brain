"""Spec §10. Must NOT immediately convert analysis into BUY/SELL — generates
competing scenarios and must permit NO_TRADE / UNCERTAIN as a legitimate
outcome, not an error case."""

from __future__ import annotations

from abc import ABC, abstractmethod

from index_option_brain.contracts.analysis import RegimeState
from index_option_brain.contracts.market_state import MarketState
from index_option_brain.contracts.scenario import Scenario


class ScenarioEngine(ABC):
    @abstractmethod
    def generate(self, state: MarketState, regime: RegimeState) -> list[Scenario]: ...
