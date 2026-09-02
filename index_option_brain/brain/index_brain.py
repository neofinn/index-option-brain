"""Spec §5. Must NOT choose options, select strikes, size positions, or
execute orders — this brain only characterizes the index itself.

Every score is a *blend* of independent measurements, and confidence falls
when they disagree or when data is missing. That is the structural answer to
spec §11's "do not build primitive single-indicator logic": no single
indicator can carry a direction here, because direction is only asserted
once the blended composite clears a threshold, and even then the Signal
Engine still has to corroborate it against breadth, options, and volatility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal

from index_option_brain.brain import indicators as ind
from index_option_brain.brain.config import IndexBrainConfig
from index_option_brain.contracts.analysis import IndexAnalysis
from index_option_brain.contracts.enums import BreakoutState, Direction, VwapRelationship
from index_option_brain.contracts.instruments import Bar
from index_option_brain.contracts.market_state import MarketState


class IndexBrain(ABC):
    @abstractmethod
    def analyze(self, state: MarketState) -> IndexAnalysis: ...


class DeterministicIndexBrain(IndexBrain):
    def __init__(self, config: IndexBrainConfig | None = None) -> None:
        self._config = config or IndexBrainConfig()

    def analyze(self, state: MarketState) -> IndexAnalysis:
        cfg = self._config
        index_state = state.index_state
        quote = index_state.quote
        spot = float(quote.ltp)

        daily = index_state.daily_bars
        intraday = index_state.intraday_bars
        evidence: list[str] = []
        invalidations: list[str] = []

        daily_closes = [float(b.close) for b in daily]
        daily_highs = [float(b.high) for b in daily]
        daily_lows = [float(b.low) for b in daily]

        # Series that include the live session, so intraday breaks and
        # momentum are measured against the price that is actually trading.
        closes_live = [*daily_closes, spot]
        highs_live = [*daily_highs, float(quote.high)]
        lows_live = [*daily_lows, float(quote.low)]

        atr_value = ind.atr(highs_live, lows_live, closes_live, cfg.atr_period)

        trend_score, trend_evidence = self._trend(closes_live, spot, atr_value, cfg)
        evidence.extend(trend_evidence)

        structure_score = ind.market_structure_score(highs_live, lows_live, cfg.swing_lookback)
        if structure_score is not None:
            evidence.append(f"Swing structure score {structure_score:+.2f} from daily pivots")

        momentum_score, momentum_evidence = self._momentum(closes_live, intraday, cfg)
        evidence.extend(momentum_evidence)

        volatility_score, volatility_evidence = self._volatility(
            highs_live, lows_live, closes_live, atr_value, cfg
        )
        evidence.extend(volatility_evidence)

        vwap_relationship, vwap_distance = self._vwap(quote.vwap, intraday, spot, atr_value, cfg)
        if vwap_relationship is not VwapRelationship.AT:
            evidence.append(
                f"Price {vwap_relationship.value.lower()} VWAP by {abs(vwap_distance):.2f} ATR"
            )

        supports, resistances = self._support_resistance(
            daily, highs_live, lows_live, spot, index_state.opening_range, cfg
        )

        breakout_state = self._breakout_state(highs_live, lows_live, spot, atr_value, cfg)
        if breakout_state is not BreakoutState.NONE:
            evidence.append(f"Breakout state: {breakout_state.value}")

        day_range_position = self._range_position(
            float(quote.low), float(quote.high), spot
        )
        opening_range_position = None
        if index_state.opening_range is not None:
            opening_range_position = self._range_position(
                float(index_state.opening_range.low),
                float(index_state.opening_range.high),
                spot,
            )

        gap_pct = None
        previous_close = float(quote.previous_close)
        if previous_close > 0:
            gap_pct = (float(quote.open) - previous_close) / previous_close * 100
            if abs(gap_pct) >= 0.5:
                evidence.append(f"Session opened with a {gap_pct:+.2f}% gap")

        composite = ind.blend(
            (trend_score, 0.4),
            (structure_score, 0.3),
            (momentum_score, 0.3),
        )
        composite = composite if composite is not None else 0.0

        direction = Direction.NEUTRAL
        if composite >= cfg.direction_threshold:
            direction = Direction.BULLISH
        elif composite <= -cfg.direction_threshold:
            direction = Direction.BEARISH

        confidence = self._confidence(
            [trend_score, structure_score, momentum_score], daily, intraday, cfg
        )

        invalidations.extend(
            self._invalidations(direction, supports, resistances, vwap_relationship, quote.vwap)
        )

        return IndexAnalysis(
            direction=direction,
            trend_score=ind.clamp(trend_score or 0.0),
            structure_score=ind.clamp(structure_score or 0.0),
            momentum_score=ind.clamp(momentum_score or 0.0),
            confidence=confidence,
            evidence=evidence,
            invalidations=invalidations,
            volatility_score=volatility_score,
            vwap_relationship=vwap_relationship,
            vwap_distance_atr=vwap_distance,
            breakout_state=breakout_state,
            support_levels=supports,
            resistance_levels=resistances,
            atr=Decimal(str(round(atr_value, 2))) if atr_value is not None else None,
            day_range_position=day_range_position,
            opening_range_position=opening_range_position,
            gap_pct=gap_pct,
        )

    def _trend(
        self,
        closes: list[float],
        spot: float,
        atr_value: float | None,
        cfg: IndexBrainConfig,
    ) -> tuple[float | None, list[str]]:
        evidence: list[str] = []
        ema_fast = ind.ema(closes, cfg.ema_fast)
        ema_slow = ind.ema(closes, cfg.ema_slow)

        separation_score = None
        if ema_fast is not None and ema_slow is not None and ema_slow != 0:
            separation = (ema_fast - ema_slow) / ema_slow
            separation_score = ind.squash(separation, cfg.ema_separation_scale)
            evidence.append(
                f"EMA{cfg.ema_fast} is {separation * 100:+.2f}% vs EMA{cfg.ema_slow}"
            )

        location_score = None
        if ema_slow is not None and atr_value:
            location_score = ind.squash((spot - ema_slow) / atr_value, 2.0)

        slope_score = None
        window = closes[-cfg.slope_period :]
        slope = ind.linreg_slope(window)
        if slope is not None and atr_value:
            slope_score = ind.squash(slope / atr_value, cfg.slope_atr_scale)
            evidence.append(f"{cfg.slope_period}-bar regression slope {slope:+.2f} pts/bar")

        return (
            ind.blend((separation_score, 0.4), (location_score, 0.25), (slope_score, 0.35)),
            evidence,
        )

    def _momentum(
        self, closes: list[float], intraday: list[Bar], cfg: IndexBrainConfig
    ) -> tuple[float | None, list[str]]:
        evidence: list[str] = []

        roc = ind.rate_of_change(closes, cfg.roc_period)
        roc_score = ind.squash(roc, cfg.roc_scale) if roc is not None else None
        if roc is not None:
            evidence.append(f"{cfg.roc_period}-bar rate of change {roc * 100:+.2f}%")

        rsi_value = ind.rsi(closes, cfg.rsi_period)
        rsi_score = None
        if rsi_value is not None:
            # Re-centred to -1..+1; one input among several, never a rule.
            rsi_score = ind.clamp((rsi_value - 50.0) / 30.0)
            evidence.append(f"RSI({cfg.rsi_period}) at {rsi_value:.1f}")

        intraday_score = None
        if len(intraday) >= cfg.min_intraday_bars:
            intraday_closes = [float(b.close) for b in intraday]
            intraday_slope = ind.linreg_slope(intraday_closes)
            reference = ind.mean(intraday_closes)
            if intraday_slope is not None and reference:
                intraday_score = ind.squash(intraday_slope / reference, 0.0004)

        return ind.blend((roc_score, 0.4), (rsi_score, 0.3), (intraday_score, 0.3)), evidence

    def _volatility(
        self,
        highs: list[float],
        lows: list[float],
        closes: list[float],
        atr_value: float | None,
        cfg: IndexBrainConfig,
    ) -> tuple[float, list[str]]:
        """Volatility as 0..1 — a magnitude, not a direction. High readings
        widen expected moves and make tight stops meaningless, which is why
        the Regime Engine reads it before classifying."""
        if atr_value is None or not closes:
            return 0.0, []
        ranges = ind.true_ranges(highs, lows, closes)
        if len(ranges) < cfg.atr_period:
            return 0.0, []
        history = [
            sum(ranges[i - cfg.atr_period : i]) / cfg.atr_period
            for i in range(cfg.atr_period, len(ranges) + 1)
        ]
        rank = ind.percentile_rank(history, atr_value)
        if rank is None:
            return 0.0, []
        evidence = [f"ATR({cfg.atr_period}) at the {rank * 100:.0f}th percentile of its own history"]
        return rank, evidence

    def _vwap(
        self,
        quote_vwap: Decimal | None,
        intraday: list[Bar],
        spot: float,
        atr_value: float | None,
        cfg: IndexBrainConfig,
    ) -> tuple[VwapRelationship, float]:
        reference = float(quote_vwap) if quote_vwap is not None else None
        if reference is None and intraday:
            reference = ind.vwap(
                [float(b.close) for b in intraday], [float(b.volume) for b in intraday]
            )
        if reference is None or not atr_value:
            return VwapRelationship.AT, 0.0

        distance_atr = (spot - reference) / atr_value
        if abs(distance_atr) < 0.1:
            return VwapRelationship.AT, distance_atr
        relationship = (
            VwapRelationship.ABOVE if distance_atr > 0 else VwapRelationship.BELOW
        )
        return relationship, distance_atr

    def _support_resistance(
        self,
        daily: list[Bar],
        highs: list[float],
        lows: list[float],
        spot: float,
        opening_range: object,
        cfg: IndexBrainConfig,
    ) -> tuple[list[Decimal], list[Decimal]]:
        """Levels from confirmed swing pivots plus the previous session's
        high/low/close — the levels that actually get defended intraday."""
        candidates: list[float] = []
        for i in ind.swing_high_indices(highs, cfg.swing_lookback):
            candidates.append(highs[i])
        for i in ind.swing_low_indices(lows, cfg.swing_lookback):
            candidates.append(lows[i])
        if daily:
            previous = daily[-1]
            candidates.extend(
                [float(previous.high), float(previous.low), float(previous.close)]
            )

        supports = sorted({c for c in candidates if c < spot}, reverse=True)
        resistances = sorted({c for c in candidates if c > spot})

        limit = cfg.support_resistance_levels
        return (
            [Decimal(str(round(level, 2))) for level in supports[:limit]],
            [Decimal(str(round(level, 2))) for level in resistances[:limit]],
        )

    def _breakout_state(
        self,
        highs: list[float],
        lows: list[float],
        spot: float,
        atr_value: float | None,
        cfg: IndexBrainConfig,
    ) -> BreakoutState:
        lookback = cfg.breakout_lookback
        if len(highs) < lookback + 2 or atr_value is None:
            return BreakoutState.NONE

        # Exclude the live bar from the range being tested, otherwise the
        # range moves with the price and nothing ever breaks out.
        range_high = max(highs[-(lookback + 1) : -1])
        range_low = min(lows[-(lookback + 1) : -1])
        buffer_points = atr_value * cfg.breakout_buffer_atr

        if spot > range_high + buffer_points:
            return BreakoutState.BREAKOUT
        if spot < range_low - buffer_points:
            return BreakoutState.BREAKDOWN

        # A session that traded clear of the range but handed it all back is a
        # failed break, which is a different future from never having broken.
        session_high = highs[-1]
        session_low = lows[-1]
        if session_high > range_high + buffer_points and spot < range_high:
            return BreakoutState.FAILED_BREAKOUT
        if session_low < range_low - buffer_points and spot > range_low:
            return BreakoutState.FAILED_BREAKDOWN
        return BreakoutState.NONE

    def _range_position(self, low: float, high: float, spot: float) -> float | None:
        """Where price sits in a range, 0 at the low and 1 at the high."""
        if high <= low:
            return None
        return ind.clamp((spot - low) / (high - low), 0.0, 1.0)

    def _confidence(
        self,
        scores: list[float | None],
        daily: list[Bar],
        intraday: list[Bar],
        cfg: IndexBrainConfig,
    ) -> float:
        present = [s for s in scores if s is not None]
        if not present:
            return 0.0

        agreement = ind.alignment(scores)
        strength = ind.mean([abs(s) for s in present]) or 0.0
        sufficiency = min(
            1.0,
            (len(daily) / cfg.min_daily_bars if cfg.min_daily_bars else 1.0),
        )
        completeness = len(present) / len(scores)
        intraday_bonus = 1.0 if len(intraday) >= cfg.min_intraday_bars else 0.85

        return ind.clamp(
            agreement * (0.35 + 0.65 * strength) * sufficiency * completeness * intraday_bonus,
            0.0,
            1.0,
        )

    def _invalidations(
        self,
        direction: Direction,
        supports: list[Decimal],
        resistances: list[Decimal],
        vwap_relationship: VwapRelationship,
        vwap_value: Decimal | None,
    ) -> list[str]:
        out: list[str] = []
        if direction is Direction.BULLISH:
            if supports:
                out.append(f"Sustained trade below {supports[0]} breaks the bullish structure")
            if vwap_value is not None and vwap_relationship is VwapRelationship.ABOVE:
                out.append(f"Loss of VWAP ({vwap_value}) removes intraday support for longs")
        elif direction is Direction.BEARISH:
            if resistances:
                out.append(f"Sustained trade above {resistances[0]} breaks the bearish structure")
            if vwap_value is not None and vwap_relationship is VwapRelationship.BELOW:
                out.append(f"Reclaim of VWAP ({vwap_value}) removes intraday support for shorts")
        return out
