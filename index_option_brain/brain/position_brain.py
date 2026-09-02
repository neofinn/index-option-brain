"""Spec §18. Core question this brain answers on every tick for every open
position: "Is the reason for entering the trade still valid?" """

from __future__ import annotations

from abc import ABC, abstractmethod

from index_option_brain.contracts.market_state import MarketState
from index_option_brain.contracts.position import Position


class PositionBrain(ABC):
    @abstractmethod
    def evaluate(self, position: Position, state: MarketState) -> Position:
        """Return an updated Position (state transition, refreshed P&L, and
        thesis-validity evidence). Must never place or modify a broker order
        directly — that is the Execution Gate / Order Manager's job."""
        ...
