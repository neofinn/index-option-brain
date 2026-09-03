"""Normalized instrument, bar, and quote contracts (spec §2).

Every data adapter, regardless of provider, must normalize its raw payloads
into these shapes before anything downstream sees them.

Price/money values are `Decimal` at the contract boundary so no rounding is
introduced by transport or storage. Indicator math converts to `float` once,
inside the brains (see `brain/indicators.py`).
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from index_option_brain.contracts.enums import OptionType


class IndexSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    name: str
    lot_size: int
    tick_size: Decimal
    strike_step: Decimal = Decimal(50)


class Bar(BaseModel):
    """One OHLCV candle. `timestamp` is the bar's open time, in UTC."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int = 0


class IndexQuote(BaseModel):
    """A point-in-time index snapshot.

    `open`/`high`/`low` describe the *current* session; `previous_close` is
    the prior session's close (the reference every percentage change in the
    system is computed against). There is deliberately no `close` field — for
    an intraday snapshot it is ambiguous, and ambiguity here would silently
    corrupt every downstream change calculation.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    timestamp: datetime
    ltp: Decimal
    open: Decimal
    high: Decimal
    low: Decimal
    previous_close: Decimal
    vwap: Decimal | None = None

    @property
    def change_pct(self) -> Decimal:
        if self.previous_close == 0:
            return Decimal(0)
        return (self.ltp - self.previous_close) / self.previous_close * 100


class ConstituentSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    name: str
    index_symbol: str
    sector: str
    weight: Decimal
    """Index weight in **percentage points**: HDFCBANK is 9.82, not 0.0982.

    Stated because both conventions look equally plausible at a call site and
    the wrong one fails silently — contributions come out 100x small, which
    reads as a flat market rather than as an error.
    """


class ConstituentQuote(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    timestamp: datetime
    ltp: Decimal
    open: Decimal
    high: Decimal
    low: Decimal
    previous_close: Decimal
    volume: int

    @property
    def change_pct(self) -> Decimal:
        if self.previous_close == 0:
            return Decimal(0)
        return (self.ltp - self.previous_close) / self.previous_close * 100


class OptionContractSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    underlying_symbol: str
    expiry: date
    strike: Decimal
    option_type: OptionType
    lot_size: int
    tick_size: Decimal
    trading_status: str = "active"

    @property
    def instrument_key(self) -> str:
        return f"{self.underlying_symbol}:{self.expiry.isoformat()}:{self.strike}:{self.option_type}"


class Greeks(BaseModel):
    model_config = ConfigDict(frozen=True)

    delta: Decimal
    gamma: Decimal
    theta: Decimal
    vega: Decimal


class OptionQuote(BaseModel):
    model_config = ConfigDict(frozen=True)

    contract: OptionContractSpec
    timestamp: datetime
    ltp: Decimal
    bid: Decimal | None
    ask: Decimal | None
    volume: int
    open_interest: int
    open_interest_change: int
    implied_volatility: Decimal | None
    greeks: Greeks | None = None

    @property
    def mid(self) -> Decimal:
        """Mid price when both sides are quoted, else LTP. Execution-facing
        code should price from this, not LTP, since a stale LTP on an
        illiquid strike is a classic source of phantom edge."""
        if self.bid is not None and self.ask is not None and self.ask >= self.bid:
            return (self.bid + self.ask) / 2
        return self.ltp

    @property
    def spread(self) -> Decimal | None:
        if self.bid is None or self.ask is None:
            return None
        return self.ask - self.bid

    @property
    def relative_spread(self) -> Decimal | None:
        """Bid-ask spread as a fraction of mid — the liquidity measure that
        actually matters when the premium is small."""
        spread = self.spread
        if spread is None:
            return None
        mid = self.mid
        if mid <= 0:
            return None
        return spread / mid


class AccountSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    available_margin: Decimal
    used_margin: Decimal
    net_equity: Decimal
