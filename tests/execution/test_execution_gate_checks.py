from index_option_brain.execution.execution_gate import ExecutionCheck, ExecutionGateResult

SPEC_SECTION_16_CHECKS = {
    "decision_valid",
    "risk_approved",
    "instrument_valid",
    "expiry_valid",
    "strike_valid",
    "lot_size_valid",
    "quantity_valid",
    "price_valid",
    "liquidity_valid",
    "spread_acceptable",
    "margin_available",
    "daily_loss_limit",
    "position_limit",
    "duplicate_order_check",
    "kill_switch",
    "market_session",
}


def test_execution_check_covers_every_mandatory_spec_check():
    assert {c.value for c in ExecutionCheck} == SPEC_SECTION_16_CHECKS


def test_gate_result_defaults_to_no_order_when_not_approved():
    result = ExecutionGateResult(approved=False, failed_checks=[ExecutionCheck.RISK_APPROVED])
    assert result.order_request is None
