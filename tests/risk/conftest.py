"""Fixtures for risk tests.

Candidates are built by hand with exact figures rather than taken from the
simulator, so every sizing assertion below is arithmetic a reader can check
without running anything.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from index_option_brain.contracts.enums import (
    Direction,
    OptionType,
    OrderSide,
    StrategyType,
)
from index_option_brain.contracts.instruments import AccountSnapshot, OptionContractSpec
from index_option_brain.contracts.risk import PortfolioState, TradeCandidate
from index_option_brain.contracts.strike import StrikeCandidate, StrikeLeg

NOW = datetime(2026, 9, 4, 6, 0, tzinfo=UTC)
EXPIRY = date(2026, 9, 10)
LOT_SIZE = 75

# A round per-lot max loss of 10,000 keeps every lot calculation in the tests
# checkable by eye.
PER_LOT_MAX_LOSS = Decimal(10000)


def contract(strike: int, option_type: OptionType = OptionType.PE) -> OptionContractSpec:
    return OptionContractSpec(
        underlying_symbol="NIFTY",
        expiry=EXPIRY,
        strike=Decimal(strike),
        option_type=option_type,
        lot_size=LOT_SIZE,
        tick_size=Decimal("0.05"),
    )


def structure(
    *,
    max_loss: Decimal = PER_LOT_MAX_LOSS,
    max_profit: Decimal | None = Decimal(2500),
    net_premium: Decimal = Decimal(-2500),
    liquidity: float = 0.85,
    spread: float = 0.01,
    strategy: StrategyType = StrategyType.PUT_CREDIT_SPREAD,
) -> StrikeCandidate:
    return StrikeCandidate(
        strategy=strategy,
        legs=[
            StrikeLeg(
                contract=contract(24600),
                side=OrderSide.SELL,
                lots=1,
                reference_price=Decimal(90),
                delta=Decimal("-0.30"),
                liquidity_score=liquidity,
            ),
            StrikeLeg(
                contract=contract(24400),
                side=OrderSide.BUY,
                lots=1,
                reference_price=Decimal("56.67"),
                delta=Decimal("-0.18"),
                liquidity_score=liquidity,
            ),
        ],
        score=0.8,
        net_premium=net_premium,
        net_delta=Decimal("9.0"),
        liquidity_score=liquidity,
        worst_relative_spread=spread,
        capital_required=max_loss,
        max_loss=max_loss,
        max_profit=max_profit,
        breakeven=[Decimal("24566.67")],
        rationale="test structure",
    )


def candidate(
    *,
    strategy: StrategyType = StrategyType.PUT_CREDIT_SPREAD,
    underlying: str = "NIFTY",
    **structure_kwargs,
) -> TradeCandidate:
    return TradeCandidate(
        state_id="state-1",
        thesis_id="thesis-1",
        signal_id="signal-1",
        scenario_id="scenario-1",
        direction=Direction.BULLISH,
        strategy=strategy,
        structure=structure(strategy=strategy, **structure_kwargs),
        underlying_symbol=underlying,
        confidence=0.85,
        evidence=["candidate evidence"],
        invalidation_conditions=["Loss of 24,500"],
    )


def account(
    *, equity: str = "2000000", available_margin: str = "1500000"
) -> AccountSnapshot:
    return AccountSnapshot(
        timestamp=NOW,
        available_margin=Decimal(available_margin),
        used_margin=Decimal(0),
        net_equity=Decimal(equity),
    )


def portfolio(*, acct: AccountSnapshot | None = None, **kwargs) -> PortfolioState:
    return PortfolioState(account=acct or account(), **kwargs)


@pytest.fixture
def default_account() -> AccountSnapshot:
    return account()


@pytest.fixture
def empty_portfolio(default_account: AccountSnapshot) -> PortfolioState:
    return portfolio(acct=default_account)


@pytest.fixture
def trade() -> TradeCandidate:
    return candidate()
