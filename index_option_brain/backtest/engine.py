"""Spec §22. The same brain must run in every RunMode (spec §26 RunMode:
LIVE/PAPER/BACKTEST/REPLAY) — the strategy must not have separate backtest
logic. This engine only swaps the data source (historical replay vs. live
adapters) and the broker (simulated fill vs. real); everything from the
event engine downward is identical code."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

from index_option_brain.contracts.position import Position


class BacktestEngine(ABC):
    @abstractmethod
    async def run(self, index_symbol: str, start: date, end: date) -> list[Position]:
        """Drive historical data through: Market State -> Event Engine ->
        Brain -> Strategy -> Risk -> Simulated Execution, and return the
        resulting closed positions."""
        ...
