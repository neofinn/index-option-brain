"""A deterministic, in-memory simulator adapter.

THIS IS NOT A LIVE ADAPTER. It exists to unblock development, unit tests, and
early paper/backtest wiring before a real broker/data-provider integration is
built. It must never be selected as the adapter for `RunMode.LIVE`
(spec §36: "Do not implement fake market data ... as if they were production
functionality. Clearly separate mocks/simulators from live adapters.").

Two properties make it useful for testing the brains rather than merely
non-empty:

* **Internally coherent.** Option premiums, deltas, gammas, thetas and vegas
  come from an actual Black-Scholes evaluation of the simulated spot, strike
  and IV — not from independent random draws. Random greeks would let a brain
  "pass" tests while being arithmetically incoherent.
* **Idempotent and reproducible.** Every value derives from the seed and the
  thing being priced, never from a mutable stream position, so reading the
  same snapshot twice returns the same numbers. A builder that fetched the
  spot twice and got two different prices would produce a MarketState that
  never existed.

The generated market's character is configurable (`daily_drift_pct`,
`daily_volatility_pct`, `breadth_bias`, `base_iv`), which is what lets tests
construct a genuine uptrend, downtrend, or range and assert that the brains
read it correctly.
"""

from __future__ import annotations

import math
import random
from datetime import UTC, date, datetime, time, timedelta
from decimal import ROUND_HALF_UP, Decimal

from index_option_brain.contracts.enums import BarInterval, OptionType
from index_option_brain.contracts.instruments import (
    AccountSnapshot,
    Bar,
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
    VolatilityDataAdapter,
)

# Lot sizes are revised periodically by the exchange; these are plausible
# values for a simulator and are not a source of truth for live trading.
_SUPPORTED_INDICES: dict[str, IndexSpec] = {
    "NIFTY": IndexSpec(
        symbol="NIFTY",
        name="Nifty 50",
        lot_size=75,
        tick_size=Decimal("0.05"),
        strike_step=Decimal(50),
    ),
    "BANKNIFTY": IndexSpec(
        symbol="BANKNIFTY",
        name="Nifty Bank",
        lot_size=30,
        tick_size=Decimal("0.05"),
        strike_step=Decimal(100),
    ),
}

_MOCK_CONSTITUENTS: dict[str, list[ConstituentSpec]] = {
    "NIFTY": [
        ConstituentSpec(
            symbol="HDFCBANK", name="HDFC Bank", index_symbol="NIFTY",
            sector="Financials", weight=Decimal("13.2"),
        ),
        ConstituentSpec(
            symbol="ICICIBANK", name="ICICI Bank", index_symbol="NIFTY",
            sector="Financials", weight=Decimal("8.9"),
        ),
        ConstituentSpec(
            symbol="RELIANCE", name="Reliance Industries", index_symbol="NIFTY",
            sector="Energy", weight=Decimal("8.4"),
        ),
        ConstituentSpec(
            symbol="INFY", name="Infosys", index_symbol="NIFTY",
            sector="IT", weight=Decimal("5.6"),
        ),
        ConstituentSpec(
            symbol="TCS", name="Tata Consultancy Services", index_symbol="NIFTY",
            sector="IT", weight=Decimal("3.9"),
        ),
        ConstituentSpec(
            symbol="BHARTIARTL", name="Bharti Airtel", index_symbol="NIFTY",
            sector="Telecom", weight=Decimal("4.3"),
        ),
        ConstituentSpec(
            symbol="LT", name="Larsen & Toubro", index_symbol="NIFTY",
            sector="Industrials", weight=Decimal("3.6"),
        ),
        ConstituentSpec(
            symbol="ITC", name="ITC", index_symbol="NIFTY",
            sector="FMCG", weight=Decimal("3.4"),
        ),
        ConstituentSpec(
            symbol="AXISBANK", name="Axis Bank", index_symbol="NIFTY",
            sector="Financials", weight=Decimal("2.9"),
        ),
        ConstituentSpec(
            symbol="MARUTI", name="Maruti Suzuki", index_symbol="NIFTY",
            sector="Auto", weight=Decimal("2.1"),
        ),
    ],
}
_MOCK_CONSTITUENTS["BANKNIFTY"] = [
    spec.model_copy(update={"index_symbol": "BANKNIFTY"})
    for spec in _MOCK_CONSTITUENTS["NIFTY"]
    if spec.sector == "Financials"
]

_RISK_FREE_RATE = 0.065
_HISTORY_DAYS = 90
_INTRADAY_BARS = 60
_INTRADAY_MINUTES = 5


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _black_scholes(
    spot: float,
    strike: float,
    years: float,
    iv: float,
    option_type: OptionType,
    rate: float = _RISK_FREE_RATE,
) -> tuple[float, float, float, float, float]:
    """Return (price, delta, gamma, theta_per_day, vega_per_iv_point).

    `iv` is a decimal fraction (0.14 == 14%). At or past expiry the option is
    worth its intrinsic value and its greeks collapse, which is the correct
    behaviour for an expiry-day simulation rather than a division by zero.
    """
    if years <= 0 or iv <= 0:
        intrinsic = (
            max(0.0, spot - strike) if option_type is OptionType.CE else max(0.0, strike - spot)
        )
        delta = 0.0
        if intrinsic > 0:
            delta = 1.0 if option_type is OptionType.CE else -1.0
        return intrinsic, delta, 0.0, 0.0, 0.0

    sqrt_t = math.sqrt(years)
    d1 = (math.log(spot / strike) + (rate + 0.5 * iv * iv) * years) / (iv * sqrt_t)
    d2 = d1 - iv * sqrt_t
    discount = math.exp(-rate * years)

    if option_type is OptionType.CE:
        price = spot * _norm_cdf(d1) - strike * discount * _norm_cdf(d2)
        delta = _norm_cdf(d1)
        theta_annual = -spot * _norm_pdf(d1) * iv / (2 * sqrt_t) - (
            rate * strike * discount * _norm_cdf(d2)
        )
    else:
        price = strike * discount * _norm_cdf(-d2) - spot * _norm_cdf(-d1)
        delta = -_norm_cdf(-d1)
        theta_annual = -spot * _norm_pdf(d1) * iv / (2 * sqrt_t) + (
            rate * strike * discount * _norm_cdf(-d2)
        )

    gamma = _norm_pdf(d1) / (spot * iv * sqrt_t)
    vega = spot * _norm_pdf(d1) * sqrt_t / 100.0
    return max(price, 0.0), delta, gamma, theta_annual / 365.0, vega


def _q(value: float, places: str = "0.05") -> Decimal:
    """Quantize to a tick-like precision."""
    step = Decimal(places)
    return (Decimal(str(value)) / step).quantize(Decimal(1), rounding=ROUND_HALF_UP) * step


class SimulatorDataAdapter(
    IndexDataAdapter,
    ConstituentDataAdapter,
    OptionsChainAdapter,
    VolatilityDataAdapter,
    AccountDataAdapter,
):
    """A single class implementing every read-side adapter interface for
    convenience in tests and local development.

    A real integration should implement each interface against the actual
    provider's API and must not be merged into one god-adapter — coverage
    differs per provider, which is the reason the interfaces are split.
    """

    def __init__(
        self,
        *,
        seed: int = 42,
        base_index_ltp: Decimal = Decimal("24500.00"),
        daily_drift_pct: float = 0.0,
        daily_volatility_pct: float = 0.7,
        intraday_drift_pct: float = 0.0,
        mean_reversion: float = 0.0,
        breadth_bias: float = 0.0,
        heavyweight_bias: float = 0.0,
        base_iv: float = 14.0,
        as_of: datetime | None = None,
        available_margin: Decimal = Decimal("500000.00"),
    ) -> None:
        self._seed = seed
        self._base_index_ltp = float(base_index_ltp)
        self._daily_drift_pct = daily_drift_pct
        self._daily_volatility_pct = daily_volatility_pct
        self._intraday_drift_pct = intraday_drift_pct
        self._mean_reversion = mean_reversion
        self._breadth_bias = breadth_bias
        self._heavyweight_bias = heavyweight_bias
        self._base_iv = base_iv
        self._as_of = as_of or datetime.now(UTC)
        self._available_margin = available_margin

        self._daily_closes = self._build_daily_path()
        self._intraday_closes = self._build_intraday_path()

    # ---------------------------------------------------------------- paths

    def _rng(self, *parts: object) -> random.Random:
        """A generator keyed by what is being produced, so results never
        depend on call order."""
        return random.Random(f"{self._seed}:" + ":".join(str(p) for p in parts))

    def _build_daily_path(self) -> list[float]:
        """A seeded price path, ending at the base price.

        `mean_reversion` pulls each step back toward the anchor, which is what
        actually produces a *range*: a pure random walk with zero drift still
        wanders into multi-day trends, so a range regime cannot be simulated
        by simply setting drift to zero.

        The path is built forwards then rebased so the current price is the
        configured one regardless of drift — otherwise a strong uptrend would
        also shift the absolute price level, and every strike-relative
        assertion would move with it.
        """
        rng = self._rng("daily")
        anchor = self._base_index_ltp
        price = anchor
        path = [price]
        for _ in range(_HISTORY_DAYS):
            shock = rng.gauss(0.0, self._daily_volatility_pct / 100.0)
            drift = self._daily_drift_pct / 100.0
            pull = self._mean_reversion * (anchor - price) / anchor
            price = max(price * (1.0 + drift + shock + pull), 1.0)
            path.append(price)
        scale = self._base_index_ltp / path[-1]
        return [p * scale for p in path]

    def _build_intraday_path(self) -> list[float]:
        rng = self._rng("intraday")
        anchor = self._daily_closes[-1]
        price = anchor
        path = []
        for _ in range(_INTRADAY_BARS):
            shock = rng.gauss(0.0, self._daily_volatility_pct / 100.0 / math.sqrt(_INTRADAY_BARS))
            drift = self._intraday_drift_pct / 100.0 / _INTRADAY_BARS
            pull = self._mean_reversion * (anchor - price) / anchor / _INTRADAY_BARS
            price = max(price * (1.0 + drift + shock + pull), 1.0)
            path.append(price)
        return path

    @property
    def _spot(self) -> float:
        return self._intraday_closes[-1]

    @property
    def _previous_close(self) -> float:
        return self._daily_closes[-1]

    # -------------------------------------------------------------- index

    async def get_index_spec(self, symbol: str) -> IndexSpec:
        try:
            return _SUPPORTED_INDICES[symbol]
        except KeyError as exc:
            raise DataAdapterError(f"unknown index symbol: {symbol}") from exc

    async def get_index_quote(self, symbol: str) -> IndexQuote:
        await self.get_index_spec(symbol)
        intraday = self._intraday_closes
        return IndexQuote(
            symbol=symbol,
            timestamp=self._as_of,
            ltp=_q(self._spot, "0.01"),
            open=_q(intraday[0], "0.01"),
            high=_q(max(intraday), "0.01"),
            low=_q(min(intraday), "0.01"),
            previous_close=_q(self._previous_close, "0.01"),
            vwap=_q(sum(intraday) / len(intraday), "0.01"),
        )

    async def get_index_bars(
        self, symbol: str, interval: BarInterval, count: int
    ) -> list[Bar]:
        await self.get_index_spec(symbol)
        if count <= 0:
            return []
        if interval is BarInterval.DAY:
            return self._daily_bars(count)
        return self._intraday_bars(interval, count)

    def _daily_bars(self, count: int) -> list[Bar]:
        closes = self._daily_closes[-count:]
        bars: list[Bar] = []
        session_date = (self._as_of - timedelta(days=len(closes))).date()
        for offset, close in enumerate(closes):
            rng = self._rng("daily-bar", offset, round(close, 4))
            previous = closes[offset - 1] if offset > 0 else close
            spread = abs(close - previous) + close * self._daily_volatility_pct / 100.0 * 0.6
            high = max(previous, close) + abs(rng.gauss(0, spread * 0.4))
            low = min(previous, close) - abs(rng.gauss(0, spread * 0.4))
            bars.append(
                Bar(
                    timestamp=datetime.combine(
                        session_date + timedelta(days=offset), time(3, 45), tzinfo=UTC
                    ),
                    open=_q(previous, "0.01"),
                    high=_q(high, "0.01"),
                    low=_q(low, "0.01"),
                    close=_q(close, "0.01"),
                    volume=rng.randint(180_000_000, 320_000_000),
                )
            )
        return bars

    def _intraday_bars(self, interval: BarInterval, count: int) -> list[Bar]:
        minutes = {
            BarInterval.MINUTE_1: 1,
            BarInterval.MINUTE_5: 5,
            BarInterval.MINUTE_15: 15,
        }.get(interval, _INTRADAY_MINUTES)
        factor = max(1, minutes // _INTRADAY_MINUTES)

        aggregated = [
            self._intraday_closes[i : i + factor]
            for i in range(0, len(self._intraday_closes), factor)
        ]
        aggregated = [chunk for chunk in aggregated if chunk][-count:]

        start = self._as_of - timedelta(minutes=minutes * len(aggregated))
        bars: list[Bar] = []
        for offset, chunk in enumerate(aggregated):
            rng = self._rng("intraday-bar", interval, offset)
            bars.append(
                Bar(
                    timestamp=start + timedelta(minutes=minutes * offset),
                    open=_q(chunk[0], "0.01"),
                    high=_q(max(chunk) * 1.0004, "0.01"),
                    low=_q(min(chunk) * 0.9996, "0.01"),
                    close=_q(chunk[-1], "0.01"),
                    volume=rng.randint(2_000_000, 9_000_000),
                )
            )
        return bars

    # -------------------------------------------------------- constituents

    async def get_constituents(self, index_symbol: str) -> list[ConstituentSpec]:
        try:
            return list(_MOCK_CONSTITUENTS[index_symbol])
        except KeyError as exc:
            raise DataAdapterError(f"no simulated constituents for: {index_symbol}") from exc

    def _heavyweight_symbols(self, count: int = 3) -> set[str]:
        specs = [
            spec for group in _MOCK_CONSTITUENTS.values() for spec in group
        ]
        ranked = sorted(specs, key=lambda s: s.weight, reverse=True)
        return {spec.symbol for spec in ranked[:count]}

    async def get_constituent_quotes(self, symbols: list[str]) -> list[ConstituentQuote]:
        index_change = (self._spot / self._previous_close - 1.0) * 100.0
        heavyweights = self._heavyweight_symbols()
        quotes: list[ConstituentQuote] = []
        for symbol in symbols:
            rng = self._rng("constituent", symbol)
            base = rng.uniform(400.0, 3600.0)
            # Each name tracks the index with its own beta plus idiosyncratic
            # noise. `breadth_bias` shifts the whole cross-section;
            # `heavyweight_bias` splits it — pushing the largest weights one
            # way and everything else the other, which is the only internally
            # consistent way to simulate a narrow, heavyweight-driven move
            # (index up while most constituents fall).
            beta = rng.uniform(0.6, 1.5)
            split = (
                self._heavyweight_bias
                if symbol in heavyweights
                else -self._heavyweight_bias
            )
            idiosyncratic = rng.gauss(self._breadth_bias + split, 0.6)
            change_pct = index_change * beta + idiosyncratic
            previous_close = base
            ltp = previous_close * (1.0 + change_pct / 100.0)
            session_open = previous_close * (1.0 + change_pct / 200.0)
            quotes.append(
                ConstituentQuote(
                    symbol=symbol,
                    timestamp=self._as_of,
                    ltp=_q(ltp, "0.05"),
                    open=_q(session_open, "0.05"),
                    high=_q(max(ltp, session_open) * 1.002, "0.05"),
                    low=_q(min(ltp, session_open) * 0.998, "0.05"),
                    previous_close=_q(previous_close, "0.05"),
                    volume=rng.randint(400_000, 9_000_000),
                )
            )
        return quotes

    # ------------------------------------------------------------- options

    async def get_available_expiries(self, underlying_symbol: str) -> list[date]:
        await self.get_index_spec(underlying_symbol)
        today = self._as_of.date()
        days_to_thursday = (3 - today.weekday()) % 7
        # On expiry day itself the contract is still live until the close, so
        # today's expiry stays in the list rather than jumping a week ahead.
        first = today + timedelta(days=days_to_thursday)
        return [first + timedelta(days=7 * i) for i in range(4)]

    async def get_option_chain(self, underlying_symbol: str, expiry: date) -> list[OptionQuote]:
        spec = await self.get_index_spec(underlying_symbol)
        spot = self._spot
        step = float(spec.strike_step)
        atm = round(spot / step) * step
        years, days = self._time_to_expiry(expiry)

        chain: list[OptionQuote] = []
        # A real index chain spans far more than a couple of sigma; too narrow
        # a simulated chain makes wide structures (condors at one sigma with
        # protective wings) unbuildable for reasons the market wouldn't impose.
        for offset in range(-20, 21):
            strike = atm + offset * step
            if strike <= 0:
                continue
            for option_type in (OptionType.CE, OptionType.PE):
                chain.append(
                    self._option_quote(spec, expiry, strike, option_type, spot, years, days)
                )
        return chain

    def _time_to_expiry(self, expiry: date) -> tuple[float, float]:
        expiry_moment = datetime.combine(expiry, time(10, 0), tzinfo=UTC)
        days = max((expiry_moment - self._as_of).total_seconds() / 86400.0, 0.0)
        return days / 365.0, days

    def _strike_iv(self, spot: float, strike: float) -> float:
        """A volatility smile with put skew, in IV percentage points."""
        moneyness = (strike - spot) / spot
        skew = -55.0 * moneyness  # OTM puts (strike < spot) price higher
        smile = 900.0 * moneyness * moneyness
        return max(4.0, self._base_iv + skew + smile)

    def _option_quote(
        self,
        spec: IndexSpec,
        expiry: date,
        strike: float,
        option_type: OptionType,
        spot: float,
        years: float,
        days: float,
    ) -> OptionQuote:
        rng = self._rng("option", expiry, strike, option_type)
        iv_pct = self._strike_iv(spot, strike)
        price, delta, gamma, theta, vega = _black_scholes(
            spot, strike, years, iv_pct / 100.0, option_type
        )
        price = max(price, 0.05)

        # Spreads widen as premium shrinks and as the strike moves away from
        # the money — the shape that makes far wings genuinely untradeable.
        distance_factor = 1.0 + abs(strike - spot) / max(spot * 0.02, 1.0)
        spread = max(0.05, price * 0.004 * distance_factor + 0.05 * distance_factor)
        bid = max(0.05, price - spread / 2)
        ask = price + spread / 2

        # Open interest peaks at ATM and again at round-number strikes, which
        # is what creates the call/put walls the Options Brain looks for.
        atm_proximity = math.exp(-((strike - spot) / (spot * 0.02)) ** 2)
        round_bonus = 2.4 if strike % (float(spec.strike_step) * 10) == 0 else 1.0
        side_bias = 1.15 if option_type is OptionType.PE and strike < spot else 1.0
        if option_type is OptionType.CE and strike > spot:
            side_bias = 1.1
        open_interest = int(
            250_000 * atm_proximity * round_bonus * side_bias + rng.uniform(5_000, 60_000)
        )

        return OptionQuote(
            contract=OptionContractSpec(
                underlying_symbol=spec.symbol,
                expiry=expiry,
                strike=Decimal(str(strike)),
                option_type=option_type,
                lot_size=spec.lot_size,
                tick_size=spec.tick_size,
            ),
            timestamp=self._as_of,
            ltp=_q(price),
            bid=_q(bid),
            ask=_q(ask),
            volume=int(open_interest * rng.uniform(0.2, 1.4)),
            open_interest=open_interest,
            open_interest_change=int(open_interest * rng.uniform(-0.15, 0.25)),
            implied_volatility=Decimal(str(round(iv_pct, 2))),
            greeks=Greeks(
                delta=Decimal(str(round(delta, 4))),
                gamma=Decimal(str(round(gamma, 6))),
                theta=Decimal(str(round(theta, 2))),
                vega=Decimal(str(round(vega, 2))),
            ),
        )

    # ---------------------------------------------------------- volatility

    async def get_india_vix(self) -> tuple[float, float]:
        rng = self._rng("vix")
        current = self._base_iv * rng.uniform(0.95, 1.08)
        previous = current * rng.uniform(0.94, 1.06)
        return round(current, 2), round(previous, 2)

    # ------------------------------------------------------------- account

    async def get_account_snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(
            timestamp=self._as_of,
            available_margin=self._available_margin,
            used_margin=Decimal("0.00"),
            net_equity=self._available_margin,
        )
