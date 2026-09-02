"""A deterministic, in-memory simulator adapter.

THIS IS NOT A LIVE ADAPTER. It exists to unblock development, unit tests, and
early paper/backtest wiring before a real broker/data-provider integration is
built. It must never be selected as the adapter for `RunMode.LIVE`
(spec §36: "Do not implement fake market data ... as if they were production
functionality. Clearly separate mocks/simulators from live adapters.").

All values are generated deterministically from a seed so tests are
reproducible.
"""

from __future__ import annotations

import random
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from index_option_brain.contracts.enums import OptionType
from index_option_brain.contracts.instruments import (
    AccountSnapshot,
    ConstituentQuote,
    ConstituentSpec,
    Greeks,
    IndexQuote,
    IndexSpec,
    OptionContractSpec,
    OptionQuote,
)
from index_option_brain.data.adapters.base import (
    AccountDataAdapter,
    ConstituentDataAdapter,
    DataAdapterError,
    IndexDataAdapter,
    OptionsChainAdapter,
)

_SUPPORTED_INDICES: dict[str, IndexSpec] = {
    "NIFTY": IndexSpec(symbol="NIFTY", name="Nifty 50", lot_size=50, tick_size=Decimal("0.05")),
    "BANKNIFTY": IndexSpec(
        symbol="BANKNIFTY", name="Nifty Bank", lot_size=15, tick_size=Decimal("0.05")
    ),
}

_MOCK_CONSTITUENTS: dict[str, list[ConstituentSpec]] = {
    "NIFTY": [
        ConstituentSpec(
            symbol="RELIANCE", name="Reliance Industries", index_symbol="NIFTY",
            sector="Energy", weight=Decimal("10.5"),
        ),
        ConstituentSpec(
            symbol="HDFCBANK", name="HDFC Bank", index_symbol="NIFTY",
            sector="Financials", weight=Decimal("9.8"),
        ),
        ConstituentSpec(
            symbol="ICICIBANK", name="ICICI Bank", index_symbol="NIFTY",
            sector="Financials", weight=Decimal("7.6"),
        ),
        ConstituentSpec(
            symbol="INFY", name="Infosys", index_symbol="NIFTY",
            sector="IT", weight=Decimal("5.9"),
        ),
    ],
}


class SimulatorDataAdapter(IndexDataAdapter, ConstituentDataAdapter, OptionsChainAdapter, AccountDataAdapter):
    """A single class implementing every read-side adapter interface for
    convenience in tests and local development.

    A real integration should instead implement each interface against the
    actual provider's API and must not be merged into one god-adapter.
    """

    def __init__(self, *, seed: int = 42, base_index_ltp: Decimal = Decimal("24500.00")) -> None:
        self._rng = random.Random(seed)
        self._base_index_ltp = base_index_ltp

    async def get_index_spec(self, symbol: str) -> IndexSpec:
        try:
            return _SUPPORTED_INDICES[symbol]
        except KeyError as exc:
            raise DataAdapterError(f"unknown index symbol: {symbol}") from exc

    async def get_index_quote(self, symbol: str) -> IndexQuote:
        await self.get_index_spec(symbol)  # validates symbol
        ltp = self._base_index_ltp + Decimal(self._rng.uniform(-50, 50)).quantize(Decimal("0.01"))
        now = datetime.now(UTC)
        return IndexQuote(
            symbol=symbol,
            timestamp=now,
            ltp=ltp,
            open=self._base_index_ltp,
            high=ltp + Decimal(25),
            low=ltp - Decimal(25),
            close=self._base_index_ltp,
            vwap=ltp,
        )

    async def get_constituents(self, index_symbol: str) -> list[ConstituentSpec]:
        try:
            return list(_MOCK_CONSTITUENTS[index_symbol])
        except KeyError as exc:
            raise DataAdapterError(f"no mock constituents for: {index_symbol}") from exc

    async def get_constituent_quotes(self, symbols: list[str]) -> list[ConstituentQuote]:
        now = datetime.now(UTC)
        quotes = []
        for symbol in symbols:
            base = Decimal(self._rng.uniform(500, 3000)).quantize(Decimal("0.01"))
            quotes.append(
                ConstituentQuote(
                    symbol=symbol,
                    timestamp=now,
                    ltp=base,
                    open=base,
                    high=base + Decimal(10),
                    low=base - Decimal(10),
                    close=base,
                    volume=self._rng.randint(100_000, 5_000_000),
                )
            )
        return quotes

    async def get_available_expiries(self, underlying_symbol: str) -> list[date]:
        await self.get_index_spec(underlying_symbol)
        today = datetime.now(UTC).date()
        days_to_thursday = (3 - today.weekday()) % 7
        next_expiry = today + timedelta(days=days_to_thursday or 7)
        return [next_expiry, next_expiry + timedelta(days=7)]

    async def get_option_chain(self, underlying_symbol: str, expiry: date) -> list[OptionQuote]:
        spec = await self.get_index_spec(underlying_symbol)
        index_quote = await self.get_index_quote(underlying_symbol)
        atm_strike = round(index_quote.ltp / 50) * 50
        now = datetime.now(UTC)
        chain: list[OptionQuote] = []
        for offset in range(-5, 6):
            strike = Decimal(atm_strike + offset * 50)
            for option_type in (OptionType.CE, OptionType.PE):
                moneyness = (strike - index_quote.ltp) if option_type == OptionType.CE else (
                    index_quote.ltp - strike
                )
                intrinsic = max(Decimal(0), -moneyness)
                time_value = Decimal(max(1.0, self._rng.uniform(5, 150)))
                contract = OptionContractSpec(
                    underlying_symbol=underlying_symbol,
                    expiry=expiry,
                    strike=strike,
                    option_type=option_type,
                    lot_size=spec.lot_size,
                    tick_size=spec.tick_size,
                )
                ltp = (intrinsic + time_value).quantize(Decimal("0.05"))
                chain.append(
                    OptionQuote(
                        contract=contract,
                        timestamp=now,
                        ltp=ltp,
                        bid=ltp - Decimal("0.5"),
                        ask=ltp + Decimal("0.5"),
                        volume=self._rng.randint(1_000, 500_000),
                        open_interest=self._rng.randint(10_000, 2_000_000),
                        open_interest_change=self._rng.randint(-50_000, 50_000),
                        implied_volatility=Decimal(self._rng.uniform(10, 25)).quantize(Decimal("0.01")),
                        greeks=Greeks(
                            delta=Decimal(self._rng.uniform(-1, 1)).quantize(Decimal("0.0001")),
                            gamma=Decimal(self._rng.uniform(0, 0.01)).quantize(Decimal("0.0001")),
                            theta=Decimal(self._rng.uniform(-20, 0)).quantize(Decimal("0.01")),
                            vega=Decimal(self._rng.uniform(0, 20)).quantize(Decimal("0.01")),
                        ),
                    )
                )
        return chain

    async def get_account_snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(
            timestamp=datetime.now(UTC),
            available_margin=Decimal("500000.00"),
            used_margin=Decimal("0.00"),
            net_equity=Decimal("500000.00"),
        )
