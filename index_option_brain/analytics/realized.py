"""Realized volatility estimators, and the premium implied vol carries over it.

The volatility risk premium — implied minus realized — is the cleanest read
on whether an option is expensive, and this module exists because the
system was computing a weak version of it.

Two defects it fixes
--------------------
**Horizon.** The old comparison put the ATM implied volatility of a
five-day option against realized volatility measured over the whole ninety
-bar window. Those describe different things: two months of calm against one
violent week is not a comparison, it is two measurements next to each
other. Realized vol is therefore measured over a window matched to the
option's own tenor, so both numbers describe the same horizon.

**Efficiency.** Close-to-close discards the intraday range, and every bar
here carries open, high, low and close. Parkinson uses the high-low range
and is roughly five times more efficient on the same sample — meaning a
20-day Parkinson estimate is about as stable as a 100-day close-to-close
one. Garman-Klass adds the open-close term and does slightly better still.
Neither handles overnight gaps, which on an Indian index are a large part
of the move; Yang-Zhang does, at the cost of needing more bars. The default
is Yang-Zhang where the sample allows and Parkinson below that, because a
short window is exactly where efficiency matters.

Signs and directions
--------------------
A **positive** premium means options are pricing more movement than the
index has delivered: expensive, which favours selling. A **negative**
premium means the market has been moving more than options are pricing:
cheap, which favours buying. This system's mandate is option buying, so the
negative case is the interesting one — and it is the case a seller-framed
rule of thumb tends to skip past.

None, never zero
----------------
Every estimator returns None when the sample cannot support it. A realized
volatility of zero is a market that did not move; an unmeasured one is a
window nobody looked at. The Volatility brain's confidence already drops
when realized is absent, and it can only do that if absence is
representable.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise

from index_option_brain.contracts.instruments import Bar

TRADING_DAYS_PER_YEAR = 252

#: Below this many returns, a variance estimate is noise.
MIN_SAMPLE = 5

#: Yang-Zhang blends three variances, each needing MIN_SAMPLE observations
#: from n-1 pairs. Set to the smallest sample where that holds rather than to
#: a comfortable margin, because the alternative below it is worse: see
#: `estimate` for the measurement that settled this.
MIN_YANG_ZHANG_SAMPLE = MIN_SAMPLE + 2


class VolatilityEstimator(StrEnum):
    CLOSE_TO_CLOSE = "close_to_close"
    PARKINSON = "parkinson"
    GARMAN_KLASS = "garman_klass"
    YANG_ZHANG = "yang_zhang"


def _annualize(daily_variance: float) -> float:
    return math.sqrt(daily_variance * TRADING_DAYS_PER_YEAR) * 100.0


def _usable(bars: Sequence[Bar]) -> list[Bar]:
    """Bars whose OHLC is internally consistent and positive.

    A bar with a high below its low, or a zero price, is a feed artifact.
    Passing it to a log-based estimator yields a NaN that propagates
    silently through every downstream comparison.
    """
    return [
        bar
        for bar in bars
        if float(bar.low) > 0
        and float(bar.open) > 0
        and float(bar.close) > 0
        and float(bar.high) >= float(bar.low)
    ]


def close_to_close(bars: Sequence[Bar]) -> float | None:
    """Annualized standard deviation of log close-to-close returns, percent."""
    usable = _usable(bars)
    if len(usable) < MIN_SAMPLE + 1:
        return None
    closes = [float(bar.close) for bar in usable]
    returns = [math.log(after / before) for before, after in pairwise(closes)]
    if len(returns) < MIN_SAMPLE:
        return None
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    return _annualize(variance)


def parkinson(bars: Sequence[Bar]) -> float | None:
    """High-low range estimator. ~5x the efficiency of close-to-close.

    Assumes no drift and no gaps: it measures how far price travelled
    *within* each session and is blind to what happened between them. On an
    Indian index that is not a small correction — over 41 real NIFTY
    sessions this read 6.76% against close-to-close's 11.23%. Use it when
    the intraday range is the question, not as a general realized estimate.
    """
    usable = _usable(bars)
    if len(usable) < MIN_SAMPLE:
        return None
    factor = 1.0 / (4.0 * math.log(2.0))
    total = sum(
        math.log(float(bar.high) / float(bar.low)) ** 2 for bar in usable
    )
    return _annualize(factor * total / len(usable))


def garman_klass(bars: Sequence[Bar]) -> float | None:
    """Range plus open-close. Slightly better than Parkinson, same blind spot.

    Also assumes no overnight gap, so it too understates a market that moves
    between sessions — 6.90% against close-to-close's 11.23% on the same 41
    NIFTY sessions.
    """
    usable = _usable(bars)
    if len(usable) < MIN_SAMPLE:
        return None
    total = 0.0
    for bar in usable:
        hl = math.log(float(bar.high) / float(bar.low))
        co = math.log(float(bar.close) / float(bar.open))
        total += 0.5 * hl * hl - (2.0 * math.log(2.0) - 1.0) * co * co
    return _annualize(max(total / len(usable), 0.0))


def yang_zhang(bars: Sequence[Bar]) -> float | None:
    """Overnight + open-to-close + Rogers-Satchell, drift-independent.

    The only estimator here that accounts for the gap between sessions,
    which is why it is the default where the sample supports it: a NIFTY
    session routinely opens 0.3-0.6 sigma away from the previous close, and
    an estimator blind to that reports a calm market that gapped every day.
    """
    usable = _usable(bars)
    if len(usable) < MIN_YANG_ZHANG_SAMPLE:
        return None

    overnight: list[float] = []
    open_to_close: list[float] = []
    rogers_satchell = 0.0
    for previous, current in pairwise(usable):
        overnight.append(math.log(float(current.open) / float(previous.close)))
        open_to_close.append(math.log(float(current.close) / float(current.open)))
        high, low = float(current.high), float(current.low)
        open_, close = float(current.open), float(current.close)
        rogers_satchell += math.log(high / close) * math.log(high / open_) + math.log(
            low / close
        ) * math.log(low / open_)

    n = len(overnight)
    if n < MIN_SAMPLE:
        return None

    def variance(values: list[float]) -> float:
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / (len(values) - 1)

    overnight_var = variance(overnight)
    open_close_var = variance(open_to_close)
    rs_var = rogers_satchell / n

    # Yang-Zhang's k minimises the estimator's variance; the constant is
    # from the original paper.
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    total = overnight_var + k * open_close_var + (1.0 - k) * rs_var
    return _annualize(max(total, 0.0))


_ESTIMATORS = {
    VolatilityEstimator.CLOSE_TO_CLOSE: close_to_close,
    VolatilityEstimator.PARKINSON: parkinson,
    VolatilityEstimator.GARMAN_KLASS: garman_klass,
    VolatilityEstimator.YANG_ZHANG: yang_zhang,
}


@dataclass(frozen=True)
class RealizedVolatility:
    """A realized volatility measurement that says how it was made.

    The estimator and window are carried because "realized volatility is
    11%" is not a fact on its own — a 20-day Parkinson number and a 90-day
    close-to-close number over the same market can differ by a third, and a
    comparison against implied is only meaningful if both sides describe the
    same horizon.
    """

    value: float
    estimator: VolatilityEstimator
    bars_used: int
    window: int

    @property
    def is_horizon_matched(self) -> bool:
        return self.bars_used >= self.window


def estimate(
    bars: Sequence[Bar],
    *,
    window: int = 20,
    estimator: VolatilityEstimator | None = None,
) -> RealizedVolatility | None:
    """Realized volatility over the most recent `window` bars, or None.

    With `estimator` unset: Yang-Zhang wherever its three variance terms can
    be computed, and close-to-close below that.

    The fallback is close-to-close and **not** Parkinson, despite Parkinson
    being the more efficient estimator, because efficiency is not the
    binding problem here — the overnight gap is. Measured over 41 real NIFTY
    sessions to 2 Sep 2026:

        window   close-to-close   Parkinson   Yang-Zhang
          10d          5.82%        5.78%        n/a
          20d          8.82%        6.32%       9.42%
          41d         11.23%        6.76%      11.56%

    Parkinson reads 6.76% where close-to-close reads 11.23% over the same
    sample — it understates by 40%, because it measures only how far price
    travelled *within* each session and an Indian index does much of its
    moving between them. Close-to-close spans the gap by construction;
    Yang-Zhang models it explicitly and tracks close-to-close closely at
    both longer windows.

    The direction of that bias is what makes it disqualifying rather than
    merely imprecise. A realized volatility biased low makes the volatility
    risk premium look high, which makes options look expensive — and this
    system's mandate is buying them. An estimator whose error argues against
    the trade you are looking for is the one to avoid.

    Parkinson and Garman-Klass stay available by explicit request, for the
    question they genuinely answer: how much did price move during the
    session, ignoring the gap.
    """
    recent = list(bars)[-window:] if window > 0 else list(bars)
    if not recent:
        return None

    chosen = estimator
    if chosen is None:
        chosen = (
            VolatilityEstimator.YANG_ZHANG
            if len(_usable(recent)) >= MIN_YANG_ZHANG_SAMPLE
            else VolatilityEstimator.CLOSE_TO_CLOSE
        )

    value = _ESTIMATORS[chosen](recent)
    if value is None or not math.isfinite(value) or value <= 0:
        return None
    return RealizedVolatility(
        value=value,
        estimator=chosen,
        bars_used=len(_usable(recent)),
        window=window,
    )


def window_for_tenor(days_to_expiry: float | None, *, floor: int = 10, cap: int = 60) -> int:
    """Sessions of history to measure against an option of this tenor.

    Calendar days are converted to sessions at 5/7 and then bounded. The
    floor exists because a five-session variance is noise, and the cap
    because a quarterly option's realized comparison should not reach back
    into a different market regime.

    The horizon match is the point: comparing a five-day option's implied
    volatility against ninety sessions of realized is what this replaces.
    """
    if days_to_expiry is None or days_to_expiry <= 0:
        return floor
    sessions = round(days_to_expiry * 5.0 / 7.0)
    return max(floor, min(cap, sessions))


@dataclass(frozen=True)
class VolatilityRiskPremium:
    """Implied minus realized, with both sides' provenance attached.

    Positive: options price more movement than the index has delivered —
    expensive, favouring sellers. Negative: the index has been moving more
    than options price — cheap, favouring buyers, which is this system's
    side.
    """

    implied: float
    realized: RealizedVolatility
    days_to_expiry: float | None

    @property
    def premium(self) -> float:
        """Implied minus realized, in volatility points."""
        return self.implied - self.realized.value

    @property
    def ratio(self) -> float:
        return self.implied / self.realized.value

    @property
    def favours_buying(self) -> bool:
        return self.premium < 0

    def score(self, *, full_scale_points: float = 6.0) -> float:
        """Premium normalised to [-1, 1].

        Scaled rather than thresholded so a mild premium contributes a small
        number instead of nothing. Six volatility points is roughly the
        spread between a quiet NIFTY week and a stressed one, so it puts a
        genuinely rich surface near the top of the range without pinning
        every ordinary day there.
        """
        raw = self.premium / full_scale_points
        return max(-1.0, min(1.0, raw))

    def describe(self) -> str:
        direction = "above" if self.premium > 0 else "below"
        return (
            f"IV {self.implied:.2f}% is {abs(self.premium):.2f} points {direction} "
            f"{self.realized.window}-session realized {self.realized.value:.2f}% "
            f"({self.realized.estimator}, {self.realized.bars_used} bars) — "
            f"premium is {'rich' if self.premium > 0 else 'cheap'}"
        )


def volatility_risk_premium(
    *,
    implied: float | None,
    bars: Sequence[Bar],
    days_to_expiry: float | None,
    estimator: VolatilityEstimator | None = None,
) -> VolatilityRiskPremium | None:
    """The premium implied vol carries over realized, horizon-matched.

    Returns None when either side is unmeasurable, rather than substituting
    a default for the missing one — a premium computed against an assumed
    realized volatility would be a statement about the assumption.
    """
    if implied is None or implied <= 0:
        return None
    realized = estimate(
        bars, window=window_for_tenor(days_to_expiry), estimator=estimator
    )
    if realized is None:
        return None
    return VolatilityRiskPremium(
        implied=implied, realized=realized, days_to_expiry=days_to_expiry
    )
