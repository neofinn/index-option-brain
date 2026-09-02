"""Canonical enumerations shared across every contract in the system.

Keeping these in one module is what makes "do not pass uncontrolled variables
between modules" (spec §3) enforceable — every brain/engine speaks the same
closed vocabulary.
"""

from __future__ import annotations

from enum import StrEnum


class Direction(StrEnum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class OptionType(StrEnum):
    CE = "CE"
    PE = "PE"


class Moneyness(StrEnum):
    ITM = "ITM"
    ATM = "ATM"
    OTM = "OTM"


class MarketRegimeType(StrEnum):
    """Spec §9."""

    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE = "RANGE"
    BREAKOUT = "BREAKOUT"
    BREAKDOWN = "BREAKDOWN"
    REVERSAL = "REVERSAL"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    EXPANSION = "EXPANSION"
    CONTRACTION = "CONTRACTION"
    EXPIRY = "EXPIRY"
    UNCERTAIN = "UNCERTAIN"


class StrategyType(StrEnum):
    """Spec §12. NO_TRADE must always be a valid, selectable member."""

    LONG_CALL = "LONG_CALL"
    LONG_PUT = "LONG_PUT"
    CALL_DEBIT_SPREAD = "CALL_DEBIT_SPREAD"
    PUT_DEBIT_SPREAD = "PUT_DEBIT_SPREAD"
    CALL_CREDIT_SPREAD = "CALL_CREDIT_SPREAD"
    PUT_CREDIT_SPREAD = "PUT_CREDIT_SPREAD"
    NEUTRAL_DEFINED_RISK = "NEUTRAL_DEFINED_RISK"
    NO_TRADE = "NO_TRADE"


class TradeDecisionType(StrEnum):
    """Spec §15 — the only primary decisions the system is allowed to make."""

    EXECUTE = "EXECUTE"
    WAIT = "WAIT"
    REJECT = "REJECT"
    EXIT = "EXIT"
    MODIFY = "MODIFY"


class MarketSessionState(StrEnum):
    """Spec §30 market state machine."""

    PRE_MARKET = "PRE_MARKET"
    OPENING = "OPENING"
    ACTIVE = "ACTIVE"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"


class TradeLifecycleState(StrEnum):
    """Spec §18/§30 position/trade state machine."""

    WATCHING = "WATCHING"
    READY = "READY"
    ENTRY_PENDING = "ENTRY_PENDING"
    ACTIVE = "ACTIVE"
    THESIS_TEST = "THESIS_TEST"
    EXIT_PENDING = "EXIT_PENDING"
    CLOSED = "CLOSED"
    RECONCILED = "RECONCILED"


class OrderLifecycleState(StrEnum):
    """Spec §30 order state machine."""

    CREATED = "CREATED"
    SUBMITTED = "SUBMITTED"
    OPEN = "OPEN"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    CANCEL_PENDING = "CANCEL_PENDING"
    CANCELLED = "CANCELLED"


class TriggerType(StrEnum):
    """Spec §4. A trigger only means "something changed; analyze it" — it must
    never directly create an order."""

    # Market triggers
    SIGNIFICANT_PRICE_MOVEMENT = "SIGNIFICANT_PRICE_MOVEMENT"
    BREAKOUT = "BREAKOUT"
    BREAKDOWN = "BREAKDOWN"
    VWAP_CROSSING = "VWAP_CROSSING"
    SUPPORT_RESISTANCE_TEST = "SUPPORT_RESISTANCE_TEST"
    OPENING_RANGE_EVENT = "OPENING_RANGE_EVENT"
    VOLATILITY_EXPANSION_CONTRACTION = "VOLATILITY_EXPANSION_CONTRACTION"
    VOLUME_ANOMALY = "VOLUME_ANOMALY"

    # Constituent triggers
    BREADTH_CHANGE = "BREADTH_CHANGE"
    MAJOR_CONSTITUENT_MOVEMENT = "MAJOR_CONSTITUENT_MOVEMENT"
    SECTOR_LEADERSHIP_CHANGE = "SECTOR_LEADERSHIP_CHANGE"
    LARGE_CONTRIBUTION_CHANGE = "LARGE_CONTRIBUTION_CHANGE"

    # Options triggers
    LARGE_OI_ADDITION = "LARGE_OI_ADDITION"
    LARGE_OI_UNWINDING = "LARGE_OI_UNWINDING"
    OI_MIGRATION = "OI_MIGRATION"
    IV_EXPANSION_COLLAPSE = "IV_EXPANSION_COLLAPSE"
    LARGE_PREMIUM_MOVEMENT = "LARGE_PREMIUM_MOVEMENT"
    GAMMA_CONCENTRATION_CHANGE = "GAMMA_CONCENTRATION_CHANGE"
    LIQUIDITY_DETERIORATION = "LIQUIDITY_DETERIORATION"

    # Time triggers
    PRE_MARKET = "PRE_MARKET"
    MARKET_OPEN = "MARKET_OPEN"
    OPENING_RANGE_COMPLETION = "OPENING_RANGE_COMPLETION"
    PERIODIC_HEARTBEAT = "PERIODIC_HEARTBEAT"
    EXPIRY_PHASE = "EXPIRY_PHASE"
    PRE_CLOSE = "PRE_CLOSE"
    END_OF_DAY = "END_OF_DAY"

    # Event triggers
    MAJOR_SCHEDULED_ECONOMIC_EVENT = "MAJOR_SCHEDULED_ECONOMIC_EVENT"
    RBI_EVENT = "RBI_EVENT"
    BUDGET_EVENT_RISK = "BUDGET_EVENT_RISK"
    INDEX_REBALANCE = "INDEX_REBALANCE"
    EXCEPTIONAL_MARKET_EVENT = "EXCEPTIONAL_MARKET_EVENT"


class FailureDomain(StrEnum):
    """Spec §29 — the conditions the failure contract must react to."""

    STALE_MARKET_DATA = "STALE_MARKET_DATA"
    BROKER_CONNECTION_LOST = "BROKER_CONNECTION_LOST"
    INCOMPLETE_OPTION_CHAIN = "INCOMPLETE_OPTION_CHAIN"
    RISK_ENGINE_FAILURE = "RISK_ENGINE_FAILURE"
    LLM_FAILURE = "LLM_FAILURE"
    REDIS_FAILURE = "REDIS_FAILURE"
    DATABASE_FAILURE = "DATABASE_FAILURE"
    STATE_RECONCILIATION_FAILURE = "STATE_RECONCILIATION_FAILURE"


class FailureAction(StrEnum):
    """Spec §29 — the only actions the failure contract may resolve to."""

    NO_NEW_TRADES = "NO_NEW_TRADES"
    NO_NEW_TRADES_AND_RECONCILE = "NO_NEW_TRADES_AND_RECONCILE"
    NO_OPTIONS_ENTRY = "NO_OPTIONS_ENTRY"
    NO_TRADE = "NO_TRADE"
    CONTINUE_WITHOUT_LLM = "CONTINUE_WITHOUT_LLM"
    FAIL_SAFE = "FAIL_SAFE"
    NO_NEW_ORDERS = "NO_NEW_ORDERS"
