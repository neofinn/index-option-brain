"""Normalized instrument and quote contracts (spec §2).

Every data adapter, regardless of provider, must normalize its raw payloads
into these shapes before anything downstream sees them.
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


class IndexQuote(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    timestamp: datetime
    ltp: Decimal
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    vwap: Decimal | None = None


class ConstituentSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    name: str
    index_symbol: str
    sector: str
    weight: Decimal


class ConstituentQuote(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    timestamp: datetime
    ltp: Decimal
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int


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


class AccountSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    available_margin: Decimal
    used_margin: Decimal
    net_equity: Decimal
