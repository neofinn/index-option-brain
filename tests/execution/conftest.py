"""Fixtures for Execution Gate tests.

The world is built by hand with exact figures so every threshold assertion is
arithmetic a reader can check. The default fixture is a NIFTY put credit
spread that passes all sixteen checks; each test then breaks exactly one thing
and asserts which check catches it.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from index_option_brain.contracts.decision import TradeDecision
from index_option_brain.contracts.enums import (
    Direction,
    MarketSessionState,
    OptionType,
    OrderSide,
    StrategyType,
    TradeDecisionType,
    TradeLifecycleState,
)
from index_option_brain.contracts.instruments import (
    AccountSnapshot,
    IndexSpec,
    OptionContractSpec,
    OptionQuote,
)
from index_option_brain.contracts.position import Position, PositionLeg
from index_option_brain.contracts.risk import PortfolioState, RiskDecision, RiskReasonCode
from index_option_brain.contracts.strike import StrikeCandidate, StrikeLeg
from index_option_brain.execution.execution_gate import ExecutionContext

IST = timezone(timedelta(hours=5, minutes=30), name="IST")

# Wednesday 02-Sep-2026, 12:00 IST — mid-session, well before the entry cutoff.
NOW = datetime(2026, 9, 2, 6, 30, tzinfo=UTC)
EXPIRY = date(2026, 9, 8)
LOT_SIZE = 75

SHORT_STRIKE = Decimal(23900)
LONG_STRIKE = Decimal(23700)
SHORT_PRICE = Decimal("114.40")
LONG_PRICE = Decimal("60.00")

# One lot of the spread: 54.40 of credit on a 200-point wing.
NET_CREDIT = (SHORT_PRICE - LONG_PRICE) * LOT_SIZE  # 4,080
MAX_LOSS = (SHORT_STRIKE - LONG_STRIKE) * LOT_SIZE - NET_CREDIT  # 10,920

INDEX_SPEC = IndexSpec(
    symbol="NIFTY",
    name="Nifty 50",
    lot_size=LOT_SIZE,
    tick_size=Decimal("0.05"),
    strike_step=Decimal(50),
)


def contract(
    strike: Decimal,
    *,
    option_type: OptionType = OptionType.PE,
    lot_size: int = LOT_SIZE,
    expiry: date = EXPIRY,
    trading_status: str = "active",
    underlying: str = "NIFTY",
) -> OptionContractSpec:
    return OptionContractSpec(
        underlying_symbol=underlying,
        expiry=expiry,
        strike=strike,
        option_type=option_type,
        lot_size=lot_size,
        tick_size=Decimal("0.05"),
        trading_status=trading_status,
    )


def quote(
    spec: OptionContractSpec,
    *,
    bid: Decimal | None,
    ask: Decimal | None,
    ltp: Decimal | None = None,
    open_interest: int = 90_000,
    volume: int = 2_500_000,
) -> OptionQuote:
    mid = (bid + ask) / 2 if bid is not None and ask is not None else Decimal(0)
    return OptionQuote(
        contract=spec,
        timestamp=NOW,
        ltp=ltp if ltp is not None else mid,
        bid=bid,
        ask=ask,
        volume=volume,
        open_interest=open_interest,
        open_interest_change=1200,
        implied_volatility=Decimal("11.4"),
    )


def structure(
    *,
    short: OptionContractSpec | None = None,
    long_: OptionContractSpec | None = None,
    short_price: Decimal = SHORT_PRICE,
    long_price: Decimal = LONG_PRICE,
    short_lots: int = 1,
    long_lots: int = 1,
    strategy: StrategyType = StrategyType.PUT_CREDIT_SPREAD,
) -> StrikeCandidate:
    return StrikeCandidate(
        strategy=strategy,
        legs=[
            StrikeLeg(
                contract=short or contract(SHORT_STRIKE),
                side=OrderSide.SELL,
                lots=short_lots,
                reference_price=short_price,
                delta=Decimal("-0.45"),
                liquidity_score=0.9,
            ),
            StrikeLeg(
                contract=long_ or contract(LONG_STRIKE),
                side=OrderSide.BUY,
                lots=long_lots,
                reference_price=long_price,
                delta=Decimal("-0.22"),
                liquidity_score=0.85,
            ),
        ],
        score=0.72,
        net_premium=-NET_CREDIT,
        net_delta=Decimal("17.25"),
        liquidity_score=0.87,
        worst_relative_spread=0.007,
        capital_required=MAX_LOSS,
        max_loss=MAX_LOSS,
        max_profit=NET_CREDIT,
        breakeven=[Decimal("23845.60")],
        rationale="bull put spread below support",
    )


def risk_decision(
    *, approved: bool = True, lots: int = 1, lot_size: int = LOT_SIZE
) -> RiskDecision:
    if not approved:
        return RiskDecision.reject(RiskReasonCode.INSUFFICIENT_RISK_BUDGET)
    return RiskDecision(
        approved=True,
        reason_codes=[RiskReasonCode.APPROVED],
        max_loss=MAX_LOSS * lots,
        quantity=lots * lot_size,
        lots=lots,
        exposure=MAX_LOSS * lots,
        margin_required=Decimal(12558),
        evidence=["Sized by the per-trade risk budget"],
    )


def decision(
    *,
    kind: TradeDecisionType = TradeDecisionType.EXECUTE,
    contracts: list[StrikeCandidate] | None = None,
    risk: RiskDecision | None = None,
    thesis_id: str = "thesis-1",
    underlying: str | None = "NIFTY",
    max_loss: Decimal | None = None,
    strategy: StrategyType = StrategyType.PUT_CREDIT_SPREAD,
) -> TradeDecision:
    structures = [structure(strategy=strategy)] if contracts is None else contracts
    return TradeDecision(
        decision_id="decision-1",
        state_id="state-1",
        thesis_id=thesis_id,
        decision=kind,
        direction=Direction.BULLISH,
        strategy=strategy,
        contracts=structures,
        entry_conditions=["Hold above 23,850"],
        target_conditions=["Decay to 50% of credit"],
        invalidation_conditions=["Acceptance below 23,700"],
        confidence=0.71,
        max_loss=MAX_LOSS if max_loss is None else max_loss,
        evidence=["Support held on the retest"],
        risk_decision=risk if risk is not None else risk_decision(),
        signal_id="signal-1",
        scenario_id="scenario-1",
        underlying_symbol=underlying,
        created_at=NOW,
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


def open_position(
    *, thesis_id: str = "thesis-other", strategy: StrategyType = StrategyType.LONG_CALL
) -> Position:
    return Position(
        position_id=f"position-{thesis_id}",
        thesis_id=thesis_id,
        state=TradeLifecycleState.ACTIVE,
        strategy=strategy,
        thesis_direction=Direction.BULLISH,
        legs=[
            PositionLeg(
                contract=contract(SHORT_STRIKE),
                side=OrderSide.SELL,
                quantity=LOT_SIZE,
                average_price=SHORT_PRICE,
            )
        ],
        max_loss=MAX_LOSS,
        opened_at=NOW,
        updated_at=NOW,
    )


def live_chain(
    *,
    short_bid: Decimal | None = Decimal("114.40"),
    short_ask: Decimal | None = Decimal("115.00"),
    long_bid: Decimal | None = Decimal("59.80"),
    long_ask: Decimal | None = Decimal("60.20"),
    short_oi: int = 90_000,
    long_oi: int = 45_000,
    short_volume: int = 2_500_000,
    long_volume: int = 900_000,
) -> list[OptionQuote]:
    return [
        quote(
            contract(SHORT_STRIKE),
            bid=short_bid,
            ask=short_ask,
            open_interest=short_oi,
            volume=short_volume,
        ),
        quote(
            contract(LONG_STRIKE),
            bid=long_bid,
            ask=long_ask,
            open_interest=long_oi,
            volume=long_volume,
        ),
    ]


def context(
    *,
    timestamp: datetime = NOW,
    session_state: MarketSessionState = MarketSessionState.ACTIVE,
    chain: list[OptionQuote] | None = None,
    acct: AccountSnapshot | None = None,
    portfolio: PortfolioState | None = None,
    kill_switch: bool = False,
    pending: frozenset[str] = frozenset(),
    index_spec: IndexSpec = INDEX_SPEC,
) -> ExecutionContext:
    resolved_account = acct or account()
    return ExecutionContext(
        timestamp=timestamp,
        session_state=session_state,
        index_spec=index_spec,
        chain=live_chain() if chain is None else chain,
        account=resolved_account,
        portfolio=portfolio or PortfolioState(account=resolved_account),
        kill_switch_engaged=kill_switch,
        pending_thesis_ids=pending,
    )


@pytest.fixture
def good_decision() -> TradeDecision:
    return decision()


@pytest.fixture
def good_context() -> ExecutionContext:
    return context()
