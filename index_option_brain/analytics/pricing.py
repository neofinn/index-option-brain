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
) -> BlackScholesResult:
    """Price one European option and return it with its greeks.

    At or past expiry — and when IV is zero — the option collapses to
    intrinsic value with zero gamma, theta and vega. That is the correct
    answer on expiry day rather than a division by zero, and expiry day is
    precisely when this code runs on a weekly.
    """
    if spot <= 0 or strike <= 0 or years <= _MIN_YEARS or iv <= _MIN_IV:
        intrinsic = _intrinsic(spot, strike, option_type)
        delta = 0.0
        if intrinsic > 0:
            delta = 1.0 if option_type is OptionType.CE else -1.0
        return BlackScholesResult(
            price=intrinsic, delta=delta, gamma=0.0, theta=0.0, vega=0.0
        )

    sqrt_t = math.sqrt(years)
    d1 = (math.log(spot / strike) + (rate + 0.5 * iv * iv) * years) / (iv * sqrt_t)
    d2 = d1 - iv * sqrt_t
    discount = math.exp(-rate * years)
    pdf_d1 = _norm_pdf(d1)

    if option_type is OptionType.CE:
        price = spot * _norm_cdf(d1) - strike * discount * _norm_cdf(d2)
        delta = _norm_cdf(d1)
        theta_annual = -spot * pdf_d1 * iv / (2 * sqrt_t) - rate * strike * discount * _norm_cdf(d2)
    else:
        price = strike * discount * _norm_cdf(-d2) - spot * _norm_cdf(-d1)
        delta = -_norm_cdf(-d1)
        theta_annual = (
            -spot * pdf_d1 * iv / (2 * sqrt_t) + rate * strike * discount * _norm_cdf(-d2)
        )

    return BlackScholesResult(
        price=max(price, 0.0),
        delta=delta,
        gamma=pdf_d1 / (spot * iv * sqrt_t),
        theta=theta_annual / 365.0,
        vega=spot * pdf_d1 * sqrt_t / 100.0,
    )


def greeks_from_iv(
    *,
    spot: float,
    strike: float,
    years: float,
    iv_percent: float,
    option_type: OptionType,
    rate: float = DEFAULT_RISK_FREE_RATE,
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
    )


def implied_volatility(
    *,
    market_price: float,
    spot: float,
    strike: float,
    years: float,
    option_type: OptionType,
    rate: float = DEFAULT_RISK_FREE_RATE,
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

    intrinsic = _intrinsic(spot, strike, option_type)
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
