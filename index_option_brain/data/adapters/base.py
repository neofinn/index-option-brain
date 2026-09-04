"""Provider-agnostic data adapter interfaces (spec §2).

The system must not be permanently tied to one broker/data provider: every
concrete adapter (Zerodha, Upstox, Angel One, a backtest replay source, ...)
implements these interfaces and hands back the normalized contracts from
`index_option_brain.contracts`, never raw provider payloads.

The interfaces are split by capability rather than by vendor, because
coverage differs: a provider may serve a live chain with Greeks but no
historical bars, or index data but no constituent weights. Splitting them
lets the system compose one working data layer out of several partial
providers, and lets the failure contract (spec §29) degrade precisely — an
incomplete chain blocks options entry without blocking everything else.

These are interfaces only. A live adapter belongs in its own module (e.g.
`data/adapters/zerodha.py`) and must never be faked as if it were production
— see `data/adapters/mock.py` for the explicitly-labeled simulator used in
tests, paper trading, and backtests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

from index_option_brain.contracts.enums import BarInterval
from index_option_brain.contracts.instruments import (
    AccountSnapshot,
    Bar,
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

    @abstractmethod
    async def get_index_bars(
        self, symbol: str, interval: BarInterval, count: int
    ) -> list[Bar]:
        """Completed bars, oldest first.

        Must NOT include the currently-forming candle: the brains treat the
        last daily bar as the previous session (PDH/PDL/PDC), and appending a
        partial bar there would silently corrupt every level derived from it.
        """
        ...


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


class VolatilityDataAdapter(ABC):
    @abstractmethod
    async def get_india_vix(self) -> tuple[float, float]:
        """Returns (current, previous_close) for India VIX."""
        ...

    async def get_india_vix_range(self) -> tuple[float, float] | None:
        """The 52-week (high, low) for India VIX, or None if unavailable.

        Optional, with a default, because not every provider publishes it —
        and a provider that does not must return None rather than a guess. It
        is worth asking for: it gives implied-volatility context immediately,
        where ranking ATM IV against its own history takes weeks of uptime.
        """
        return None


class AccountDataAdapter(ABC):
    @abstractmethod
    async def get_account_snapshot(self) -> AccountSnapshot: ...
