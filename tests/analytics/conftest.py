"""Fixtures shared by the analytics tests."""

from __future__ import annotations

from decimal import Decimal

import pytest

from index_option_brain.brain import structures
from index_option_brain.contracts.enums import OptionType
from tests.brain.test_structures import PRICES, _quote


@pytest.fixture
def view() -> structures.ChainView:
    """The same hand-built chain the structure tests use, so a cost assertion
    and an economics assertion are talking about the same instrument."""
    chain = [
        _quote(strike, option_type)
        for strike in PRICES
        for option_type in (OptionType.CE, OptionType.PE)
    ]
    built = structures.ChainView.from_chain(chain, Decimal(24500))
    assert built is not None
    return built
