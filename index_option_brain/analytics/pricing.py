"""Black-Scholes pricing and greeks.

This is production analytics, not test scaffolding, because **no Indian data
source supplies option greeks**. NSE's chain publishes implied volatility and
nothing else; broker APIs publish neither. Delta, gamma, theta and vega are
therefore computed here from the live premium and IV, and the Strike Engine's
delta-fit ranking depends on it.

Indian index options are European-style and cash-settled, which is what makes
plain Black-Scholes appropriate rather than an American-exercise model.

Conventions:
  * `iv` is a decimal fraction (0.14 == 14%). Feed data that reports IV in
    percentage points through `/ 100` first.
  * `years` is calendar time to expiry. Option value decays over the
    calendar, not the trading session — using trading days here would
    understate weekend decay on a Tuesday-expiry weekly.
  * `theta` is per calendar day; `vega` is per one-point move in IV
    (i.e. per 1%), which is how both are quoted on a trading desk.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from index_option_brain.contracts.enums import OptionType

# The RBI repo rate is the usual reference for INR-denominated carry. It is a
# parameter rather than a constant because it moves, and because a wrong rate
# quietly biases every delta on the board.
DEFAULT_RISK_FREE_RATE = 0.065

_MIN_YEARS = 1e-9
_MIN_IV = 1e-9


@dataclass(frozen=True)
class BlackScholesResult:
    price: float
    delta: float
    gamma: float
    theta: float
    """Per calendar day."""
    vega: float
    """Per one IV point (1%)."""

    @property
    def is_degenerate(self) -> bool:
        """True at or past expiry, or with no volatility: the option is worth
        its intrinsic value and its sensitivities have collapsed."""
        return self.gamma == 0.0 and self.vega == 0.0


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _intrinsic(spot: float, strike: float, option_type: OptionType) -> float:
    if option_type is OptionType.CE:
        return max(0.0, spot - strike)
    return max(0.0, strike - spot)


def price_option(
    *,
    spot: float,
    strike: float,
    years: float,
    iv: float,
    option_type: OptionType,
    rate: float = DEFAULT_RISK_FREE_RATE,
    dividend_yield: float = 0.0,
) -> BlackScholesResult:
    """Price one European option and return it with its greeks.

    At or past expiry — and when IV is zero — the option collapses to
    intrinsic value with zero gamma, theta and vega. That is the correct
    answer on expiry day rather than a division by zero, and expiry day is
    precisely when this code runs on a weekly.

    `dividend_yield` is the continuous carry subtracted from `rate`, so the
    model prices off a forward of `spot * exp((rate - dividend_yield) *
    years)`. Left at 0.0 the forward is `spot * exp(rate * years)` and every
    result is identical to the plain Black-Scholes form.

    Why it is here: NIFTY options are cash-settled European contracts whose
    market forward is set by the futures, and that forward is not
    `spot * exp(rate * years)`. On 3 Sep 2026 the synthetic forward from
    put-call parity stood 43.7 points above spot against a carry of 22.4 —
    a 21-point excess. Pricing off spot alone pushes that discrepancy into
    the volatility, which is why the same chain solved to a call IV of 10.6%
    and a put IV of 8.7% at the same strike. With the carry supplied, the two
    agree to 0.01 points. The distortion is not cosmetic: it moved the 24,200
    call's delta from 0.290 to 0.304, across a 0.30 floor.
    """
    if spot <= 0 or strike <= 0 or years <= _MIN_YEARS or iv <= _MIN_IV:
        intrinsic = _intrinsic(spot * math.exp(-dividend_yield * years), strike, option_type)
        delta = 0.0
        if intrinsic > 0:
            delta = 1.0 if option_type is OptionType.CE else -1.0
        return BlackScholesResult(
            price=intrinsic, delta=delta, gamma=0.0, theta=0.0, vega=0.0
        )

    sqrt_t = math.sqrt(years)
    carry = rate - dividend_yield
    d1 = (math.log(spot / strike) + (carry + 0.5 * iv * iv) * years) / (iv * sqrt_t)
    d2 = d1 - iv * sqrt_t
    discount = math.exp(-rate * years)
    # The spot leg is discounted by the carry the holder gives up.
    carry_discount = math.exp(-dividend_yield * years)
    pdf_d1 = _norm_pdf(d1)
    carried_spot = spot * carry_discount

    if option_type is OptionType.CE:
        price = carried_spot * _norm_cdf(d1) - strike * discount * _norm_cdf(d2)
        delta = carry_discount * _norm_cdf(d1)
        theta_annual = (
            -carried_spot * pdf_d1 * iv / (2 * sqrt_t)
            + dividend_yield * carried_spot * _norm_cdf(d1)
            - rate * strike * discount * _norm_cdf(d2)
        )
    else:
        price = strike * discount * _norm_cdf(-d2) - carried_spot * _norm_cdf(-d1)
        delta = -carry_discount * _norm_cdf(-d1)
        theta_annual = (
            -carried_spot * pdf_d1 * iv / (2 * sqrt_t)
            - dividend_yield * carried_spot * _norm_cdf(-d1)
            + rate * strike * discount * _norm_cdf(-d2)
        )

    return BlackScholesResult(
        price=max(price, 0.0),
        delta=delta,
        gamma=carry_discount * pdf_d1 / (spot * iv * sqrt_t),
        theta=theta_annual / 365.0,
        vega=carried_spot * pdf_d1 * sqrt_t / 100.0,
    )


def greeks_from_iv(
    *,
    spot: float,
    strike: float,
    years: float,
    iv_percent: float,
    option_type: OptionType,
    rate: float = DEFAULT_RISK_FREE_RATE,
    dividend_yield: float = 0.0,
) -> BlackScholesResult:
    """Convenience wrapper for data feeds that quote IV in percentage points.

    This is the entry point the live NSE adapter uses: the chain gives an IV
    like `11.43`, and everything downstream needs the greeks that implies.
    """
    return price_option(
        spot=spot,
        strike=strike,
        years=years,
        iv=iv_percent / 100.0,
        option_type=option_type,
        rate=rate,
        dividend_yield=dividend_yield,
    )


def implied_volatility(
    *,
    market_price: float,
    spot: float,
    strike: float,
    years: float,
    option_type: OptionType,
    rate: float = DEFAULT_RISK_FREE_RATE,
    dividend_yield: float = 0.0,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> float | None:
    """Recover IV from a premium by bisection, or None if it cannot be found.

    Needed when a feed gives a price but no IV — several broker APIs do
    exactly that. Bisection rather than Newton-Raphson because vega collapses
    for deep-out-of-the-money and near-expiry options, where Newton diverges
    precisely when it is most needed.

    Returns None rather than a fabricated number when the price is below
    intrinsic value (stale or crossed data) or outside the search bracket.
    """
    if market_price <= 0 or years <= _MIN_YEARS or spot <= 0 or strike <= 0:
        return None

    intrinsic = _intrinsic(spot * math.exp(-dividend_yield * years), strike, option_type)
    if market_price < intrinsic - tolerance:
        # Below intrinsic there is no volatility that explains the price.
        return None

    low, high = 1e-4, 5.0  # 0.01% to 500% IV
    for _ in range(max_iterations):
        mid = (low + high) / 2
        price = price_option(
            spot=spot,
            strike=strike,
            years=years,
            iv=mid,
            option_type=option_type,
            rate=rate,
            dividend_yield=dividend_yield,
        ).price
        difference = price - market_price
        if abs(difference) < tolerance:
            return mid
        if difference > 0:
            high = mid
        else:
            low = mid

    resolved = (low + high) / 2
    # Refuse to report a value that has merely hit the edge of the bracket.
    if resolved <= 1e-3 or resolved >= 4.99:
        return None
    return resolved


@dataclass(frozen=True)
class ForwardEstimate:
    """The market's forward for one expiry, recovered from put-call parity.

    `strikes_used` is carried because the estimate is only as good as the
    number of strikes with a two-sided book on both legs, and one strike is a
    quote, not a measurement.
    """

    forward: float
    spot: float
    years: float
    rate: float
    strikes_used: int

    @property
    def basis(self) -> float:
        """Forward minus spot, in index points."""
        return self.forward - self.spot

    @property
    def carry_basis(self) -> float:
        """Basis implied by the risk-free rate alone, in index points."""
        return self.spot * (math.exp(self.rate * self.years) - 1.0)

    @property
    def excess_basis(self) -> float:
        """Basis beyond pure carry, in index points.

        This is the number that carries information. Pure carry is mechanical;
        what the market pays *above* it is positioning. On 3 Sep 2026 this
        went from -24 points at the previous close to +22 intraday, a 45-point
        swing in one session.
        """
        return self.basis - self.carry_basis

    @property
    def dividend_yield(self) -> float:
        """The continuous carry that makes Black-Scholes reproduce this forward.

        Feed it to `price_option` and the model's forward becomes the market's,
        which is what makes a call's IV and a put's IV agree at the same strike.
        """
        if self.years <= _MIN_YEARS or self.spot <= 0 or self.forward <= 0:
            return 0.0
        return self.rate - math.log(self.forward / self.spot) / self.years


def forward_from_parity(
    *,
    pairs: Sequence[tuple[float, float, float]],
    spot: float,
    years: float,
    rate: float = DEFAULT_RISK_FREE_RATE,
    max_strikes: int = 6,
) -> ForwardEstimate | None:
    """Recover the market forward from put-call parity, or None.

    `pairs` is `(strike, call_price, put_price)`, prices being mids on a
    two-sided book. Parity gives `F = K + (C - P) * exp(rate * years)` at
    every strike; the estimate averages the `max_strikes` nearest the money,
    where both legs are liquid and the difference is least noisy.

    Returns None rather than a number when nothing usable is supplied: an
    invented forward would silently reprice the whole chain.
    """
    if years <= _MIN_YEARS or spot <= 0 or not pairs:
        return None

    usable = [
        (abs(strike - spot), strike, call, put)
        for strike, call, put in pairs
        if strike > 0 and call > 0 and put > 0
    ]
    if not usable:
        return None
    usable.sort()

    grown = math.exp(rate * years)
    nearest = usable[:max_strikes]
    forwards = [strike + (call - put) * grown for _, strike, call, put in nearest]
    return ForwardEstimate(
        forward=sum(forwards) / len(forwards),
        spot=spot,
        years=years,
        rate=rate,
        strikes_used=len(forwards),
    )
