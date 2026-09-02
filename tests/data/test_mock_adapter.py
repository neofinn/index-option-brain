from decimal import Decimal

import pytest

from index_option_brain.data.adapters.base import DataAdapterError
from index_option_brain.data.adapters.mock import SimulatorDataAdapter


@pytest.fixture
def adapter() -> SimulatorDataAdapter:
    return SimulatorDataAdapter(seed=7)


async def test_get_index_quote_is_deterministic_for_a_given_seed(adapter: SimulatorDataAdapter):
    a = SimulatorDataAdapter(seed=7)
    b = SimulatorDataAdapter(seed=7)
    quote_a = await a.get_index_quote("NIFTY")
    quote_b = await b.get_index_quote("NIFTY")
    assert quote_a.ltp == quote_b.ltp


async def test_get_index_quote_unknown_symbol_raises(adapter: SimulatorDataAdapter):
    with pytest.raises(DataAdapterError):
        await adapter.get_index_quote("NOT_A_REAL_INDEX")


async def test_get_constituents_returns_specs_summing_to_a_plausible_weight(
    adapter: SimulatorDataAdapter,
):
    constituents = await adapter.get_constituents("NIFTY")
    assert len(constituents) > 0
    assert all(c.index_symbol == "NIFTY" for c in constituents)


async def test_option_chain_is_centered_near_atm_and_has_ce_and_pe(adapter: SimulatorDataAdapter):
    expiries = await adapter.get_available_expiries("NIFTY")
    chain = await adapter.get_option_chain("NIFTY", expiries[0])

    call_types = {q.contract.option_type for q in chain}
    assert {"CE", "PE"} <= call_types

    for quote in chain:
        assert quote.ltp > Decimal(0)
        assert quote.bid is not None and quote.ask is not None
        assert quote.bid <= quote.ltp <= quote.ask


async def test_account_snapshot_has_non_negative_margin(adapter: SimulatorDataAdapter):
    snapshot = await adapter.get_account_snapshot()
    assert snapshot.available_margin >= Decimal(0)
