import pytest

from index_option_brain.contracts.enums import FailureAction, FailureDomain
from index_option_brain.risk.failure_policy import FailurePolicy


@pytest.mark.parametrize(
    ("domain", "expected"),
    [
        (FailureDomain.STALE_MARKET_DATA, FailureAction.NO_NEW_TRADES),
        (FailureDomain.BROKER_CONNECTION_LOST, FailureAction.NO_NEW_TRADES_AND_RECONCILE),
        (FailureDomain.INCOMPLETE_OPTION_CHAIN, FailureAction.NO_OPTIONS_ENTRY),
        (FailureDomain.RISK_ENGINE_FAILURE, FailureAction.NO_TRADE),
        (FailureDomain.LLM_FAILURE, FailureAction.CONTINUE_WITHOUT_LLM),
        (FailureDomain.REDIS_FAILURE, FailureAction.FAIL_SAFE),
        (FailureDomain.DATABASE_FAILURE, FailureAction.NO_NEW_TRADES),
        (FailureDomain.STATE_RECONCILIATION_FAILURE, FailureAction.NO_NEW_ORDERS),
    ],
)
def test_every_failure_domain_has_a_mapped_safe_action(domain, expected):
    assert FailurePolicy.action_for(domain) is expected


def test_every_failure_domain_is_covered():
    for domain in FailureDomain:
        # Must not raise KeyError — every domain resolves to a safe action.
        FailurePolicy.action_for(domain)
