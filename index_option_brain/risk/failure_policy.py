"""Spec §29 — Failure Contract, expressed as an explicit, testable mapping
rather than scattered conditionals. Every failure domain resolves to exactly
one safe action; there is no domain whose failure is silently ignored."""

from __future__ import annotations

from index_option_brain.contracts.enums import FailureAction, FailureDomain

FAILURE_ACTIONS: dict[FailureDomain, FailureAction] = {
    FailureDomain.STALE_MARKET_DATA: FailureAction.NO_NEW_TRADES,
    FailureDomain.BROKER_CONNECTION_LOST: FailureAction.NO_NEW_TRADES_AND_RECONCILE,
    FailureDomain.INCOMPLETE_OPTION_CHAIN: FailureAction.NO_OPTIONS_ENTRY,
    FailureDomain.RISK_ENGINE_FAILURE: FailureAction.NO_TRADE,
    FailureDomain.LLM_FAILURE: FailureAction.CONTINUE_WITHOUT_LLM,
    FailureDomain.REDIS_FAILURE: FailureAction.FAIL_SAFE,
    FailureDomain.DATABASE_FAILURE: FailureAction.NO_NEW_TRADES,
    FailureDomain.STATE_RECONCILIATION_FAILURE: FailureAction.NO_NEW_ORDERS,
}


class FailurePolicy:
    @staticmethod
    def action_for(domain: FailureDomain) -> FailureAction:
        return FAILURE_ACTIONS[domain]
