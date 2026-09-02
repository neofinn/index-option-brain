"""Provider-agnostic data adapter interfaces (spec §2).

The system must not be permanently tied to one broker/data provider: every
concrete adapter (Zerodha, Upstox, Angel One, a backtest replay source, ...)
implements these interfaces and hands back the normalized contracts from
`index_option_brain.contracts`, never raw provider payloads.

These are interfaces only. A live adapter belongs in its own module (e.g.
`data/adapters/zerodha.py`) and must never be faked as if it were production
— see `data/adapters/mock.py` for the explicitly-labeled simulator used in
tests, paper trading, and backtests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

from index_option_brain.contracts.instruments import (
    AccountSnapshot,
    ConstituentQuote,
    ConstituentSpec,
    IndexQuote,
    IndexSpec,
    OptionQuote,
)


class DataAdapterError(RuntimeError):
    """Raised by an adapter when it cannot produce a normalized quote/spec.

    Per the failure contract (spec §29), callers must treat this as a signal
    to withhold new trades, not as something to silently paper over.
    """


class IndexDataAdapter(ABC):
    @abstractmethod
    async def get_index_spec(self, symbol: str) -> IndexSpec: ...

    @abstractmethod
    async def get_index_quote(self, symbol: str) -> IndexQuote: ...


class ConstituentDataAdapter(ABC):
    @abstractmethod
    async def get_constituents(self, index_symbol: str) -> list[ConstituentSpec]: ...

    @abstractmethod
    async def get_constituent_quotes(self, symbols: list[str]) -> list[ConstituentQuote]: ...


class OptionsChainAdapter(ABC):
    @abstractmethod
    async def get_available_expiries(self, underlying_symbol: str) -> list[date]: ...

    @abstractmethod
    async def get_option_chain(
        self, underlying_symbol: str, expiry: date
    ) -> list[OptionQuote]: ...


class AccountDataAdapter(ABC):
    @abstractmethod
    async def get_account_snapshot(self) -> AccountSnapshot: ...
