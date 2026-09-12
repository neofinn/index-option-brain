"""Pure indicator functions shared by the brains.

Every function here is deterministic, side-effect free, and takes plain
`float` sequences — no MarketState, no config, no I/O. That keeps the same
code exercisable in LIVE, PAPER, BACKTEST, and REPLAY (spec §22: "The same
brain must run in every mode") and makes the numeric core independently
testable from the judgement layered on top of it.

Insufficient data returns `None` rather than a fabricated value. A brain that
receives `None` must lower its confidence, never substitute a default — a
made-up indicator reading is indistinguishable from a real one downstream,
which is exactly the failure mode spec §36 warns about.

Series are ordered oldest-first; the most recent observation is last.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from decimal import Decimal
from itertools import pairwise


def to_floats(values: Sequence[Decimal]) -> list[float]:
    return [float(v) for v in values]


def mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def stdev(values: Sequence[float]) -> float | None:
    """Sample standard deviation (n-1). Needs at least two observations."""
    if len(values) < 2:
        return None
    avg = sum(values) / len(values)
    variance = sum((v - avg) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def sma(values: Sequence[float], period: int) -> float | None:
    if period <= 0 or len(values) < period:
        return None
    window = values[-period:]
    return sum(window) / period


def ema_series(values: Sequence[float], period: int) -> list[float]:
    """Exponential moving average, seeded with the first observation."""
    if period <= 0 or not values:
        return []
    k = 2.0 / (period + 1)
    out = [values[0]]
    for value in values[1:]:
        out.append(value * k + out[-1] * (1 - k))
    return out


def ema(values: Sequence[float], period: int) -> float | None:
    if period <= 0 or len(values) < period:
        return None
    return ema_series(values, period)[-1]


def rsi(values: Sequence[float], period: int = 14) -> float | None:
    """Simple-average (Cutler's) RSI over the last `period` changes.

    Returned as 0..100. Callers should re-center it (`rsi/50 - 1`) before
    blending it with the -1..+1 scores used everywhere else.
    """
    if period <= 0 or len(values) < period + 1:
        return None
    window = values[-(period + 1) :]
    gains = 0.0
    losses = 0.0
    for previous, current in pairwise(window):
        delta = current - previous
        gains += max(delta, 0.0)
        losses += max(-delta, 0.0)
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def true_ranges(
    highs: Sequence[float], lows: Sequence[float], closes: Sequence[float]
) -> list[float]:
    n = min(len(highs), len(lows), len(closes))
    if n < 2:
        return []
    out = []
    for i in range(1, n):
        previous_close = closes[i - 1]
        out.append(
            max(
                highs[i] - lows[i],
                abs(highs[i] - previous_close),
                abs(lows[i] - previous_close),
            )
        )
    return out


def atr(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> float | None:
    ranges = true_ranges(highs, lows, closes)
    if period <= 0 or len(ranges) < period:
        return None
    return sum(ranges[-period:]) / period


def rate_of_change(values: Sequence[float], period: int) -> float | None:
    """Fractional change over `period` observations (0.01 == +1%)."""
    if period <= 0 or len(values) < period + 1:
        return None
    past = values[-(period + 1)]
    if past == 0:
        return None
    return (values[-1] - past) / past


def linreg_slope(values: Sequence[float]) -> float | None:
    """Least-squares slope in price units per observation."""
    n = len(values)
    if n < 2:
        return None
    mean_x = (n - 1) / 2
    mean_y = sum(values) / n
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in enumerate(values))
    denominator = sum((x - mean_x) ** 2 for x in range(n))
    if denominator == 0:
        return None
    return numerator / denominator


def vwap(prices: Sequence[float], volumes: Sequence[float]) -> float | None:
    n = min(len(prices), len(volumes))
    if n == 0:
        return None
    total_volume = sum(volumes[:n])
    if total_volume <= 0:
        return mean(prices[:n])
    return sum(p * v for p, v in zip(prices[:n], volumes[:n], strict=True)) / total_volume


def percentile_rank(history: Sequence[float], value: float) -> float | None:
    """Fraction of `history` at or below `value`, as 0..1."""
    if not history:
        return None
    at_or_below = sum(1 for h in history if h <= value)
    return at_or_below / len(history)


def swing_high_indices(highs: Sequence[float], lookback: int = 2) -> list[int]:
    """Indices of bars whose high strictly exceeds `lookback` neighbours on
    both sides."""
    if lookback <= 0:
        return []
    out = []
    for i in range(lookback, len(highs) - lookback):
        pivot = highs[i]
        if all(pivot > highs[j] for j in range(i - lookback, i)) and all(
            pivot > highs[j] for j in range(i + 1, i + lookback + 1)
        ):
            out.append(i)
    return out


def swing_low_indices(lows: Sequence[float], lookback: int = 2) -> list[int]:
    if lookback <= 0:
        return []
    out = []
    for i in range(lookback, len(lows) - lookback):
        pivot = lows[i]
        if all(pivot < lows[j] for j in range(i - lookback, i)) and all(
            pivot < lows[j] for j in range(i + 1, i + lookback + 1)
        ):
            out.append(i)
    return out


def market_structure_score(
    highs: Sequence[float], lows: Sequence[float], lookback: int = 2
) -> float | None:
    """Swing structure as -1..+1.

    +1 is a clean sequence of higher highs *and* higher lows, -1 lower highs
    *and* lower lows, 0 a mixed/compressing structure. Returns None when
    there aren't two confirmed swings on either side to compare.
    """
    high_pivots = swing_high_indices(highs, lookback)
    low_pivots = swing_low_indices(lows, lookback)
    parts: list[float] = []
    if len(high_pivots) >= 2:
        parts.append(1.0 if highs[high_pivots[-1]] > highs[high_pivots[-2]] else -1.0)
    if len(low_pivots) >= 2:
        parts.append(1.0 if lows[low_pivots[-1]] > lows[low_pivots[-2]] else -1.0)
    return mean(parts)


def squash(value: float, scale: float) -> float:
    """Map an unbounded magnitude onto -1..+1, where `scale` is the value that
    maps to ~0.76. Smooth and monotonic, so small changes never produce a
    discontinuous jump in a score."""
    if scale <= 0:
        return 0.0
    return math.tanh(value / scale)


def clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def normalized_hhi(weights: Sequence[float]) -> float | None:
    """Herfindahl concentration of |weights|, rescaled so 0 is perfectly even
    and 1 is a single dominant member."""
    magnitudes = [abs(w) for w in weights]
    total = sum(magnitudes)
    n = len(magnitudes)
    if n == 0 or total <= 0:
        return None
    if n == 1:
        return 1.0
    hhi = sum((m / total) ** 2 for m in magnitudes)
    floor = 1.0 / n
    return clamp((hhi - floor) / (1.0 - floor), 0.0, 1.0)


def alignment(scores: Sequence[float | None]) -> float:
    """Agreement among signed scores, as 0..1.

    1.0 means every non-null score points the same way; 0.0 means they cancel
    exactly. This is what separates "three domains independently agree" from
    "one strong reading dragging two contradictions along", which is the
    distinction the Signal Engine needs to not fire on a single indicator.
    """
    values = [s for s in scores if s is not None]
    if not values:
        return 0.0
    total_magnitude = sum(abs(v) for v in values)
    if total_magnitude == 0:
        return 0.0
    return abs(sum(values)) / total_magnitude


def blend(*weighted: tuple[float | None, float]) -> float | None:
    """Weighted mean over the components that are present.

    Missing components are dropped and the remaining weights renormalized, so
    a partially-observable state degrades smoothly instead of being scored as
    if the missing inputs were zero.
    """
    present = [(value, weight) for value, weight in weighted if value is not None and weight > 0]
    if not present:
        return None
    total_weight = sum(weight for _, weight in present)
    if total_weight <= 0:
        return None
    return sum(value * weight for value, weight in present) / total_weight
