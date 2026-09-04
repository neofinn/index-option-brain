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


def _index_quote(ltp: str = "24500", previous_close: str = "24450") -> IndexQuote:
    return IndexQuote(
        symbol="NIFTY",
        timestamp=datetime.now(UTC),
        ltp=Decimal(ltp),
        open=Decimal(24460),
        high=Decimal(24550),
        low=Decimal(24400),
        previous_close=Decimal(previous_close),
    )


def _state(**overrides) -> MarketState:
    defaults = {
        "timestamp": datetime.now(UTC),
        "index_state": IndexState(quote=_index_quote()),
        "constituent_state": ConstituentState(),
        "sector_state": SectorState(),
        "options_state": OptionsState(),
        "volatility_state": VolatilityState(),
    }
    return MarketState(**{**defaults, **overrides})


def test_market_state_builds_with_empty_sub_states():
    state = _state()
    assert state.market_regime is None
    assert state.analysis is None
    assert state.active_events == []
    assert state.position_state.positions == []
    assert state.state_id


def test_market_state_is_frozen():
    state = _state()
    with pytest.raises(ValidationError):
        state.timestamp = datetime.now(UTC)  # type: ignore[misc]


def test_advancing_state_returns_a_new_instance():
    """The pipeline advances state by copy, so one stage can never mutate the
    snapshot another stage is reading."""
    from index_option_brain.contracts.analysis import RegimeState
    from index_option_brain.contracts.enums import MarketRegimeType

    original = _state()
    advanced = original.with_regime(
        RegimeState(regime=MarketRegimeType.TREND_UP, confidence=0.8)
    )
    assert original.market_regime is None
    assert advanced.market_regime is not None
    assert advanced.state_id == original.state_id


def test_change_pct_is_computed_against_the_previous_close():
    quote = _index_quote(ltp="24500", previous_close="24000")
    assert quote.change_pct == pytest.approx(Decimal("2.0833"), abs=Decimal("0.001"))


def test_change_pct_of_a_zero_previous_close_does_not_divide_by_zero():
    quote = _index_quote(ltp="24500", previous_close="0")
    assert quote.change_pct == 0


def test_spot_and_symbol_shortcuts_read_from_the_index_quote():
    state = _state()
    assert state.spot == Decimal(24500)
    assert state.index_symbol == "NIFTY"


def test_option_contract_instrument_key_is_stable():
    contract = OptionContractSpec(
        underlying_symbol="NIFTY",
        expiry=date(2026, 9, 25),
        strike=Decimal(24500),
        option_type=OptionType.CE,
        lot_size=75,
        tick_size=Decimal("0.05"),
    )
    assert contract.instrument_key == "NIFTY:2026-09-25:24500:CE"
