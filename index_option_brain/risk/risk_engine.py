"""Spec §14. Risk has absolute authority over trade authorization. If risk
fails or cannot be evaluated: REJECT. No AI/agent may override this
(spec §23, §35) — `IntelligenceProvider` has no method that returns a
RiskDecision or bypasses one."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from index_option_brain.contracts.decision import TradeDecision
from index_option_brain.contracts.instruments import AccountSnapshot
from index_option_brain.contracts.position import PositionState
from index_option_brain.contracts.risk import RiskDecision


class Portfolio(Protocol):
    """The subset of portfolio/account state the Risk Engine must weigh:
    exposure, margin, daily loss, concentration, and open-position limits."""

    position_state: PositionState
    account: AccountSnapshot
    daily_realized_pnl: object
    open_position_count: int


class RiskEngine(ABC):
    @abstractmethod
    def authorize(self, trade: TradeDecision, account: AccountSnapshot, portfolio: Portfolio) -> RiskDecision: ...
