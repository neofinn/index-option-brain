"""Spec §11. A Signal is NOT an order. Do not build primitive
single-indicator logic such as `RSI > 50 -> BUY` here — this engine exists to
combine evidence across every domain (index, breadth, sectors, options, IV,
Greeks, market structure, time/expiry context) against the competing
scenarios it is handed."""

from __future__ import annotations

from abc import ABC, abstractmethod

from index_option_brain.contracts.market_state import MarketState
from index_option_brain.contracts.scenario import Scenario
from index_option_brain.contracts.signal import Signal


class SignalEngine(ABC):
    @abstractmethod
    def evaluate(self, state: MarketState, scenarios: list[Scenario]) -> Signal: ...
