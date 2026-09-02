"""Spec §16. Only the deterministic execution layer may talk to the broker,
and only after every mandatory check below passes. If any mandatory check
fails: NO ORDER — there is no override path, LLM or otherwise."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from index_option_brain.contracts.decision import TradeDecision
from index_option_brain.contracts.order import OrderRequest


class ExecutionCheck(StrEnum):
    DECISION_VALID = "decision_valid"
    RISK_APPROVED = "risk_approved"
    INSTRUMENT_VALID = "instrument_valid"
    EXPIRY_VALID = "expiry_valid"
    STRIKE_VALID = "strike_valid"
    LOT_SIZE_VALID = "lot_size_valid"
    QUANTITY_VALID = "quantity_valid"
    PRICE_VALID = "price_valid"
    LIQUIDITY_VALID = "liquidity_valid"
    SPREAD_ACCEPTABLE = "spread_acceptable"
    MARGIN_AVAILABLE = "margin_available"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    POSITION_LIMIT = "position_limit"
    DUPLICATE_ORDER_CHECK = "duplicate_order_check"
    KILL_SWITCH = "kill_switch"
    MARKET_SESSION = "market_session"


class ExecutionGateResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    approved: bool
    failed_checks: list[ExecutionCheck] = Field(default_factory=list)
    order_request: OrderRequest | None = None


class ExecutionGate(ABC):
    @abstractmethod
    def validate(self, decision: TradeDecision) -> ExecutionGateResult:
        """Run every ExecutionCheck against `decision`. Must return
        approved=False (and never an OrderRequest) unless every mandatory
        check in ExecutionCheck passes."""
        ...
