from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from index_option_brain.contracts.enums import OptionType
from index_option_brain.contracts.instruments import IndexQuote, OptionContractSpec
from index_option_brain.contracts.market_state import (
    ConstituentState,
    IndexState,
    MarketState,
    OptionsState,
    SectorState,
    VolatilityState,
)


def _index_quote() -> IndexQuote:
    return IndexQuote(
        symbol="NIFTY",
        timestamp=datetime.now(UTC),
        ltp=Decimal(24500),
        open=Decimal(24450),
        high=Decimal(24550),
        low=Decimal(24400),
        close=Decimal(24450),
    )


def test_market_state_builds_with_empty_sub_states():
    state = MarketState(
        timestamp=datetime.now(UTC),
        index_state=IndexState(quote=_index_quote()),
        constituent_state=ConstituentState(),
        sector_state=SectorState(),
        options_state=OptionsState(),
        volatility_state=VolatilityState(),
    )
    assert state.market_regime is None
    assert state.active_events == []
    assert state.position_state.positions == []


def test_market_state_is_frozen():
    state = MarketState(
        timestamp=datetime.now(UTC),
        index_state=IndexState(quote=_index_quote()),
        constituent_state=ConstituentState(),
        sector_state=SectorState(),
        options_state=OptionsState(),
        volatility_state=VolatilityState(),
    )
    with pytest.raises(ValidationError):
        state.timestamp = datetime.now(UTC)  # type: ignore[misc]


def test_option_contract_instrument_key_is_stable():
    contract = OptionContractSpec(
        underlying_symbol="NIFTY",
        expiry=date(2026, 9, 25),
        strike=Decimal(24500),
        option_type=OptionType.CE,
        lot_size=50,
        tick_size=Decimal("0.05"),
    )
    assert contract.instrument_key == "NIFTY:2026-09-25:24500:CE"
