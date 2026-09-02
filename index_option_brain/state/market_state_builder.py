"""The market-state engine (spec §1, §3): assembles adapter output into a
single, immutable MarketState.

This is the only place that knows about individual data adapters —
everything downstream (event engine, brains, risk, execution) consumes
MarketState and nothing else. It is also where derived-but-observational
quantities live (realized volatility, ATM IV, sector aggregates, session
state): they are *measurements*, not judgements, so they belong in state
rather than in a brain.

Timestamps are UTC internally (spec §26). Session boundaries are evaluated
against IST because that is what defines the Indian trading day.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from itertools import pairwise

from index_option_brain.contracts.enums import BarInterval, MarketSessionState, OptionType
from index_option_brain.contracts.instruments import (
    Bar,
    ConstituentQuote,
    ConstituentSpec,
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
from index_option_brain.data.adapters.base import (
    ConstituentDataAdapter,
    IndexDataAdapter,
    OptionsChainAdapter,
    VolatilityDataAdapter,
)

IST = timezone(timedelta(hours=5, minutes=30))

_MARKET_OPEN = time(9, 15)
_OPENING_ENDS = time(9, 30)
_CLOSING_BEGINS = time(15, 0)
_MARKET_CLOSE = time(15, 30)

_TRADING_DAYS_PER_YEAR = 252


class IvHistoryStore(ABC):
    """Recent ATM IV observations, needed to rank IV against its own history.

    This is deliberately an injected dependency rather than something the
    builder accumulates privately: spec §36 requires no hidden global state,
    and in BACKTEST/REPLAY the history must come from the replayed timeline,
    not from whatever this process happens to have seen.
    """

    @abstractmethod
    def record(self, symbol: str, atm_iv: float) -> None: ...

    @abstractmethod
    def history(self, symbol: str) -> list[float]: ...


class InMemoryIvHistoryStore(IvHistoryStore):
    def __init__(self, max_observations: int = 250) -> None:
        self._max = max_observations
        self._data: dict[str, list[float]] = {}

    def record(self, symbol: str, atm_iv: float) -> None:
        series = self._data.setdefault(symbol, [])
        series.append(atm_iv)
        if len(series) > self._max:
            del series[: len(series) - self._max]

    def history(self, symbol: str) -> list[float]:
        return list(self._data.get(symbol, []))


class MarketStateBuilder:
    def __init__(
        self,
        index_adapter: IndexDataAdapter,
        constituent_adapter: ConstituentDataAdapter,
        options_adapter: OptionsChainAdapter,
        volatility_adapter: VolatilityDataAdapter | None = None,
        iv_history: IvHistoryStore | None = None,
        *,
        daily_bar_count: int = 90,
        intraday_bar_count: int = 75,
        intraday_interval: BarInterval = BarInterval.MINUTE_5,
        opening_range_minutes: int = 15,
    ) -> None:
        self._index_adapter = index_adapter
        self._constituent_adapter = constituent_adapter
        self._options_adapter = options_adapter
        self._volatility_adapter = volatility_adapter
        self._iv_history = iv_history
        self._daily_bar_count = daily_bar_count
        self._intraday_bar_count = intraday_bar_count
        self._intraday_interval = intraday_interval
        self._opening_range_minutes = opening_range_minutes

    async def build(
        self, index_symbol: str, options_expiry: date | None = None
    ) -> MarketState:
        spec = await self._index_adapter.get_index_spec(index_symbol)
        quote = await self._index_adapter.get_index_quote(index_symbol)
        daily_bars = await self._index_adapter.get_index_bars(
            index_symbol, BarInterval.DAY, self._daily_bar_count
        )
        intraday_bars = await self._index_adapter.get_index_bars(
            index_symbol, self._intraday_interval, self._intraday_bar_count
        )

        expiries = await self._options_adapter.get_available_expiries(index_symbol)
        expiry = options_expiry or (expiries[0] if expiries else None)
        chain = (
            await self._options_adapter.get_option_chain(index_symbol, expiry)
            if expiry is not None
            else []
        )

        constituent_specs = await self._constituent_adapter.get_constituents(index_symbol)
        constituent_quotes = await self._constituent_adapter.get_constituent_quotes(
            [spec_.symbol for spec_ in constituent_specs]
        )

        weights = {s.symbol: float(s.weight) for s in constituent_specs}
        sectors = {s.symbol: s.sector for s in constituent_specs}

        vix = vix_previous = None
        if self._volatility_adapter is not None:
            vix, vix_previous = await self._volatility_adapter.get_india_vix()

        atm_iv = self._atm_iv(chain, quote.ltp)
        if atm_iv is not None and self._iv_history is not None:
            self._iv_history.record(index_symbol, atm_iv)
        history = self._iv_history.history(index_symbol) if self._iv_history else []

        days_to_expiry = (
            self._days_to_expiry(expiry, quote.timestamp) if expiry is not None else None
        )

        return MarketState(
            timestamp=quote.timestamp,
            session_state=self.session_state(quote.timestamp),
            index_state=IndexState(
                quote=quote,
                spec=spec,
                intraday_bars=intraday_bars,
                daily_bars=daily_bars,
                opening_range=self._opening_range(intraday_bars),
            ),
            constituent_state=ConstituentState(
                quotes=constituent_quotes, weights=weights, sectors=sectors
            ),
            sector_state=self._sector_state(constituent_specs, constituent_quotes),
            options_state=OptionsState(
                chain=chain, expiry=expiry, available_expiries=expiries
            ),
            volatility_state=VolatilityState(
                india_vix=vix,
                india_vix_previous_close=vix_previous,
                realized_volatility=self._realized_volatility(daily_bars),
                atm_iv=atm_iv,
                atm_iv_history=history,
                days_to_expiry=days_to_expiry,
            ),
        )

    def session_state(self, moment: datetime) -> MarketSessionState:
        local = moment.astimezone(IST).time()
        if local < _MARKET_OPEN:
            return MarketSessionState.PRE_MARKET
        if local < _OPENING_ENDS:
            return MarketSessionState.OPENING
        if local < _CLOSING_BEGINS:
            return MarketSessionState.ACTIVE
        if local < _MARKET_CLOSE:
            return MarketSessionState.CLOSING
        return MarketSessionState.CLOSED

    def _opening_range(self, intraday_bars: list[Bar]) -> OpeningRange | None:
        """High/low of the first N minutes of the session.

        Only bars from the current session count, so a series that starts
        mid-session yields no opening range rather than a misleading one
        computed from whatever the earliest available bars happen to be.
        """
        if not intraday_bars:
            return None

        session_start_bars = [
            bar
            for bar in intraday_bars
            if bar.timestamp.astimezone(IST).time() < self._opening_range_end()
            and bar.timestamp.astimezone(IST).time() >= _MARKET_OPEN
        ]
        if not session_start_bars:
            return None

        last_local = intraday_bars[-1].timestamp.astimezone(IST).time()
        return OpeningRange(
            high=max(bar.high for bar in session_start_bars),
            low=min(bar.low for bar in session_start_bars),
            completed=last_local >= self._opening_range_end(),
        )

    def _opening_range_end(self) -> time:
        # Any date works: only the resulting time-of-day is used.
        end = datetime.combine(date(2000, 1, 1), _MARKET_OPEN) + timedelta(
            minutes=self._opening_range_minutes
        )
        return end.time()

    def _sector_state(
        self, specs: list[ConstituentSpec], quotes: list[ConstituentQuote]
    ) -> SectorState:
        by_symbol = {q.symbol: q for q in quotes}
        returns: dict[str, float] = {}
        weights: dict[str, float] = {}
        weighted: dict[str, float] = {}

        for spec in specs:
            quote = by_symbol.get(spec.symbol)
            if quote is None:
                continue
            weight = float(spec.weight)
            weights[spec.sector] = weights.get(spec.sector, 0.0) + weight
            weighted[spec.sector] = weighted.get(spec.sector, 0.0) + weight * float(
                quote.change_pct
            )

        for sector, total_weight in weights.items():
            if total_weight > 0:
                returns[sector] = weighted[sector] / total_weight

        return SectorState(sector_returns=returns, sector_weights=weights)

    def _atm_iv(self, chain: list[OptionQuote], spot: Decimal) -> float | None:
        """Average of the CE and PE implied volatilities at the nearest
        strike — using one side alone would inherit that side's skew."""
        if not chain:
            return None
        strikes = {q.contract.strike for q in chain}
        if not strikes:
            return None
        atm_strike = min(strikes, key=lambda s: abs(s - spot))
        ivs = [
            float(q.implied_volatility)
            for q in chain
            if q.contract.strike == atm_strike
            and q.implied_volatility is not None
            and q.contract.option_type in (OptionType.CE, OptionType.PE)
        ]
        if not ivs:
            return None
        return sum(ivs) / len(ivs)

    def _days_to_expiry(self, expiry: date, now: datetime) -> float:
        expiry_moment = datetime.combine(expiry, time(15, 30), tzinfo=IST)
        return max((expiry_moment - now).total_seconds() / 86400.0, 0.0)

    def _realized_volatility(self, daily_bars: list[Bar]) -> float | None:
        """Annualized close-to-close realized volatility, in percent.

        Trading days are used here (252) because realized volatility is
        measured from observed sessions — unlike the expected-move
        calculation, which decays over calendar time.
        """
        if len(daily_bars) < 3:
            return None
        closes = [float(bar.close) for bar in daily_bars]
        returns = [
            math.log(current / previous)
            for previous, current in pairwise(closes)
            if previous > 0 and current > 0
        ]
        if len(returns) < 2:
            return None
        average = sum(returns) / len(returns)
        variance = sum((r - average) ** 2 for r in returns) / (len(returns) - 1)
        return math.sqrt(variance) * math.sqrt(_TRADING_DAYS_PER_YEAR) * 100.0
