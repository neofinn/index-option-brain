"""Fixtures for event detection.

States are built by hand from a small base rather than from the simulator, so
each test changes exactly one thing and the reader can see what the detector
is reacting to.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from index_option_brain.contracts.analysis import (
    AnalysisBundle,
    ConstituentAnalysis,
    IndexAnalysis,
    OptionsAnalysis,
    VolatilityAnalysis,
)
from index_option_brain.contracts.enums import (
    Direction,
    IvRegime,
    MarketSessionState,
    OptionType,
)
from index_option_brain.contracts.instruments import (
    Bar,
    ConstituentQuote,
    Greeks,
    IndexQuote,
    OptionContractSpec,
    OptionQuote,
)
from index_option_brain.contracts.market_state import (
    ConstituentState,
    IndexState,
    MarketState,
    OpeningRange,
    OptionsState,
    SectorState,
    VolatilityState,
)

NOW = datetime(2026, 9, 2, 6, 30, tzinfo=UTC)
EXPIRY = date(2026, 9, 8)
SPOT = Decimal(23900)


def bar(close: str, *, offset: int = 0, volume: int = 100_000) -> Bar:
    value = Decimal(close)
    return Bar(
        timestamp=NOW + timedelta(minutes=5 * offset),
        open=value,
        high=value + 20,
        low=value - 20,
        close=value,
        volume=volume,
    )


def option(
    strike: int,
    option_type: OptionType,
    *,
    bid: str = "100.00",
    ask: str = "101.00",
    oi: int = 50_000,
    oi_change: int = 0,
    gamma: str = "0.0015",
    iv: str = "11.4",
) -> OptionQuote:
    return OptionQuote(
        contract=OptionContractSpec(
            underlying_symbol="NIFTY",
            expiry=EXPIRY,
            strike=Decimal(strike),
            option_type=option_type,
            lot_size=75,
            tick_size=Decimal("0.05"),
        ),
        timestamp=NOW,
        ltp=Decimal(ask),
        bid=Decimal(bid),
        ask=Decimal(ask),
        volume=1_000_000,
        open_interest=oi,
        open_interest_change=oi_change,
        implied_volatility=Decimal(iv),
        greeks=Greeks(
            delta=Decimal("0.5"),
            gamma=Decimal(gamma),
            theta=Decimal(-10),
            vega=Decimal(12),
        ),
    )


def default_chain() -> list[OptionQuote]:
    return [
        option(23800, OptionType.CE, oi=40_000),
        option(23900, OptionType.CE, oi=90_000),
        option(24000, OptionType.CE, oi=30_000),
        option(23800, OptionType.PE, oi=35_000),
        option(23900, OptionType.PE, oi=85_000),
        option(24000, OptionType.PE, oi=25_000),
    ]


def analysis(
    *,
    atr: str | None = "150",
    support: list[str] | None = None,
    resistance: list[str] | None = None,
    breadth: float = 0.2,
    concentration: float = 0.3,
) -> AnalysisBundle:
    return AnalysisBundle(
        index=IndexAnalysis(
            direction=Direction.NEUTRAL,
            trend_score=0.0,
            structure_score=0.0,
            momentum_score=0.0,
            confidence=0.6,
            atr=Decimal(atr) if atr is not None else None,
            support_levels=[Decimal(level) for level in (support or [])],
            resistance_levels=[Decimal(level) for level in (resistance or [])],
        ),
        constituents=ConstituentAnalysis(
            breadth_score=breadth,
            participation_score=0.6,
            leadership_score=0.0,
            concentration_score=concentration,
            confidence=0.6,
        ),
        options=OptionsAnalysis(
            call_pressure=0.3,
            put_pressure=0.3,
            oi_structure_score=0.0,
            iv_score=0.0,
            liquidity_score=0.8,
            confidence=0.7,
        ),
        volatility=VolatilityAnalysis(
            regime=IvRegime.NORMAL,
            expected_move=400,
            iv_score=0.0,
            expansion_score=0.0,
            confidence=0.6,
            days_to_expiry=6.0,
        ),
    )


def state(
    *,
    ltp: str = "23900",
    previous_close: str = "23900",
    open_: str = "23900",
    high: str = "23950",
    low: str = "23850",
    vwap: str | None = None,
    timestamp: datetime = NOW,
    session: MarketSessionState = MarketSessionState.ACTIVE,
    chain: list[OptionQuote] | None = None,
    india_vix: float | None = 11.34,
    atm_iv: float | None = 10.0,
    realized_vol: float | None = 12.0,
    days_to_expiry: float | None = 6.0,
    daily_bars: list[Bar] | None = None,
    intraday_bars: list[Bar] | None = None,
    opening_range: OpeningRange | None = None,
    with_analysis: AnalysisBundle | None = None,
    constituents: list[ConstituentQuote] | None = None,
    weights: dict[str, float] | None = None,
    sector_returns: dict[str, float] | None = None,
) -> MarketState:
    return MarketState(
        timestamp=timestamp,
        session_state=session,
        index_state=IndexState(
            quote=IndexQuote(
                symbol="NIFTY",
                timestamp=timestamp,
                ltp=Decimal(ltp),
                open=Decimal(open_),
                high=Decimal(high),
                low=Decimal(low),
                previous_close=Decimal(previous_close),
                vwap=Decimal(vwap) if vwap is not None else None,
            ),
            daily_bars=daily_bars if daily_bars is not None else [bar("23880")],
            intraday_bars=intraday_bars or [],
            opening_range=opening_range,
        ),
        constituent_state=ConstituentState(
            quotes=constituents or [], weights=weights or {}
        ),
        sector_state=SectorState(sector_returns=sector_returns or {}),
        options_state=OptionsState(
            chain=default_chain() if chain is None else chain,
            expiry=EXPIRY,
            available_expiries=[EXPIRY],
        ),
        volatility_state=VolatilityState(
            india_vix=india_vix,
            india_vix_previous_close=11.49,
            realized_volatility=realized_vol,
            atm_iv=atm_iv,
            days_to_expiry=days_to_expiry,
        ),
        analysis=with_analysis,
    )


def constituent(symbol: str, ltp: str, previous_close: str = "1000") -> ConstituentQuote:
    return ConstituentQuote(
        symbol=symbol,
        timestamp=NOW,
        ltp=Decimal(ltp),
        open=Decimal(previous_close),
        high=Decimal(ltp),
        low=Decimal(previous_close),
        previous_close=Decimal(previous_close),
        volume=1_000_000,
    )


@pytest.fixture
def base() -> MarketState:
    return state()
