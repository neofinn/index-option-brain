"""Detectors for each trigger type (spec §4).

Each detector is a small pure function of `(previous, current, config)`
returning the events it found. Written that way for three reasons: a trigger
can be tested on its own, a missing detector is visible as a gap in the
registry rather than as silence, and no detector can accidentally suppress
another by returning early.

Two rules hold for every one of them.

**A detector never fires on unmeasured data.** If India VIX is `None` in
either snapshot, no volatility trigger fires — it does not treat the missing
value as zero and report a collapse. The same discipline as the brains: the
absence of a measurement is not a measurement.

**A detector never fires on the first tick.** With no previous state there is
nothing to compare, and a system that reported "significant price movement"
the moment it started would wake the pipeline on its own arrival. Time
triggers are the exception, because a session boundary is a fact about the
clock rather than a change between snapshots.

A trigger only ever means "something changed; analyze it". None of these may
create an order, and nothing here reads a live price to act on.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal

from index_option_brain.contracts.enums import (
    MarketSessionState,
    OptionType,
    TriggerType,
)
from index_option_brain.contracts.events import Event
from index_option_brain.contracts.instruments import OptionQuote
from index_option_brain.contracts.market_state import MarketState
from index_option_brain.events.config import TriggerEngineConfig

Detector = Callable[[MarketState | None, MarketState, TriggerEngineConfig], list[Event]]


def _event(
    trigger: TriggerType,
    state: MarketState,
    *,
    significance: float,
    **payload: object,
) -> Event:
    """Build one event.

    `significance` is set by the detector because only the detector knows the
    scale of the thing it measured — a 3% index move and a 3% change in one
    strike's open interest are not comparable magnitudes. Deciding whether
    that score is worth acting on belongs to the significance filter.
    """
    return Event(
        event_id=uuid.uuid4().hex[:12],
        trigger_type=trigger,
        timestamp=state.timestamp,
        payload={"state_id": state.state_id, **payload},
        significance_score=max(0.0, min(1.0, significance)),
    )


def _scale(magnitude: float, threshold: float) -> float:
    """How far past its threshold a measurement is, saturating at 1.

    Anything at exactly the threshold scores 0.5, and twice the threshold
    scores 1.0. The point is that "just over the line" and "enormous" should
    not produce the same score, because the filter's floor is the only thing
    standing between a quiet tick and a full analysis.
    """
    if threshold <= 0:
        return 1.0
    return min(1.0, abs(magnitude) / (2.0 * threshold))


# --------------------------------------------------------------------- price


def detect_price_movement(
    previous: MarketState | None, current: MarketState, cfg: TriggerEngineConfig
) -> list[Event]:
    if previous is None:
        return []
    before = previous.index_state.quote.ltp
    now = current.index_state.quote.ltp
    if before <= 0:
        return []
    move = now - before

    # ATR-relative where possible: a fixed number of index points means
    # something different at 12% volatility than at 30%.
    atr = _atr(previous)
    if atr is not None and atr > 0:
        magnitude = float(abs(move) / atr)
        threshold = cfg.price_move_atr
        basis = "atr"
    else:
        magnitude = float(abs(move) / before * 100)
        threshold = cfg.price_move_pct
        basis = "pct"

    if magnitude < threshold:
        return []
    return [
        _event(
            TriggerType.SIGNIFICANT_PRICE_MOVEMENT,
            current,
            significance=_scale(magnitude, threshold),
            move=float(move),
            magnitude=round(magnitude, 4),
            basis=basis,
            from_price=float(before),
            to_price=float(now),
        )
    ]


def detect_exceptional_move(
    previous: MarketState | None, current: MarketState, cfg: TriggerEngineConfig
) -> list[Event]:
    """A move large enough to be a different kind of day.

    Measured against the previous close rather than against the last snapshot,
    because the thing that makes a day exceptional is where it has got to, not
    how fast the last tick was.
    """
    quote = current.index_state.quote
    change = abs(float(quote.change_pct))
    if change < cfg.exceptional_move_pct:
        return []
    return [
        _event(
            TriggerType.EXCEPTIONAL_MARKET_EVENT,
            current,
            significance=_scale(change, cfg.exceptional_move_pct),
            change_pct=round(float(quote.change_pct), 4),
            reason="index move from the previous close is beyond a normal session",
        )
    ]


def detect_breakout(
    previous: MarketState | None, current: MarketState, cfg: TriggerEngineConfig
) -> list[Event]:
    """A cross of the previous session's high or low.

    Uses the last *completed* daily bar. The adapter contract guarantees the
    forming candle is excluded, which is what makes these levels the previous
    session's rather than a mix of yesterday and today.
    """
    if previous is None:
        return []
    bars = current.index_state.daily_bars
    if not bars:
        return []
    reference = bars[-1]
    before = previous.index_state.quote.ltp
    now = current.index_state.quote.ltp

    events: list[Event] = []
    if before <= reference.high < now:
        events.append(
            _event(
                TriggerType.BREAKOUT,
                current,
                significance=0.7,
                level=float(reference.high),
                level_name="previous session high",
                price=float(now),
            )
        )
    if before >= reference.low > now:
        events.append(
            _event(
                TriggerType.BREAKDOWN,
                current,
                significance=0.7,
                level=float(reference.low),
                level_name="previous session low",
                price=float(now),
            )
        )
    return events


def detect_opening_range_event(
    previous: MarketState | None, current: MarketState, cfg: TriggerEngineConfig
) -> list[Event]:
    if previous is None:
        return []
    opening = current.index_state.opening_range
    if opening is None or not opening.completed:
        return []
    before = previous.index_state.quote.ltp
    now = current.index_state.quote.ltp

    events: list[Event] = []
    if before <= opening.high < now:
        events.append(
            _event(
                TriggerType.OPENING_RANGE_EVENT,
                current,
                significance=0.65,
                direction="above",
                level=float(opening.high),
            )
        )
    if before >= opening.low > now:
        events.append(
            _event(
                TriggerType.OPENING_RANGE_EVENT,
                current,
                significance=0.65,
                direction="below",
                level=float(opening.low),
            )
        )
    return events


def detect_vwap_crossing(
    previous: MarketState | None, current: MarketState, cfg: TriggerEngineConfig
) -> list[Event]:
    """Fires only when VWAP is actually published.

    NSE's public feed does not publish an index VWAP, so this stays silent
    there rather than crossing a fabricated line.
    """
    if previous is None:
        return []
    before_vwap = previous.index_state.quote.vwap
    now_vwap = current.index_state.quote.vwap
    if before_vwap is None or now_vwap is None:
        return []
    before_side = previous.index_state.quote.ltp >= before_vwap
    now_side = current.index_state.quote.ltp >= now_vwap
    if before_side == now_side:
        return []
    return [
        _event(
            TriggerType.VWAP_CROSSING,
            current,
            significance=0.55,
            direction="above" if now_side else "below",
            vwap=float(now_vwap),
            price=float(current.index_state.quote.ltp),
        )
    ]


def detect_level_test(
    previous: MarketState | None, current: MarketState, cfg: TriggerEngineConfig
) -> list[Event]:
    """Price approaching a level the *last* analysis identified.

    The levels come from `previous.analysis`, which is the right source: the
    trigger engine runs before analysis in a cycle, so the only levels
    available are the ones the previous cycle published. That also means a
    level is only tested once it has been reasoned about.
    """
    if previous is None or previous.analysis is None:
        return []
    index = previous.analysis.index
    atr = index.atr
    if atr is None or atr <= 0:
        return []
    tolerance = atr * Decimal(str(cfg.level_test_atr))
    price = current.index_state.quote.ltp

    events: list[Event] = []
    for level, kind in [
        *((level, "support") for level in index.support_levels),
        *((level, "resistance") for level in index.resistance_levels),
    ]:
        distance = abs(price - level)
        if distance > tolerance:
            continue
        # Only a *new* approach counts. Sitting within tolerance for twenty
        # ticks is one test, not twenty.
        if abs(previous.index_state.quote.ltp - level) <= tolerance:
            continue
        events.append(
            _event(
                TriggerType.SUPPORT_RESISTANCE_TEST,
                current,
                significance=0.6,
                level=float(level),
                kind=kind,
                distance=float(distance),
            )
        )
    return events


def detect_volume_anomaly(
    previous: MarketState | None, current: MarketState, cfg: TriggerEngineConfig
) -> list[Event]:
    """Latest bar's volume against the recent baseline.

    Silent when the feed reports no volume at all, which is the case for NSE's
    index snapshot — a baseline of zeros would make every bar an anomaly.
    """
    bars = current.index_state.intraday_bars
    if len(bars) < cfg.min_bars_for_volume_baseline + 1:
        return []
    baseline_bars = bars[-(cfg.min_bars_for_volume_baseline + 1) : -1]
    baseline = sum(bar.volume for bar in baseline_bars) / len(baseline_bars)
    if baseline <= 0:
        return []
    latest = bars[-1].volume
    ratio = latest / baseline
    if ratio < cfg.volume_anomaly_ratio:
        return []
    return [
        _event(
            TriggerType.VOLUME_ANOMALY,
            current,
            significance=_scale(ratio, cfg.volume_anomaly_ratio),
            ratio=round(ratio, 3),
            volume=latest,
            baseline=round(baseline, 1),
        )
    ]


# ---------------------------------------------------------------- volatility


def detect_volatility_change(
    previous: MarketState | None, current: MarketState, cfg: TriggerEngineConfig
) -> list[Event]:
    if previous is None:
        return []
    events: list[Event] = []

    before_vix = previous.volatility_state.india_vix
    now_vix = current.volatility_state.india_vix
    if before_vix is not None and now_vix is not None and before_vix > 0:
        change = (now_vix - before_vix) / before_vix * 100
        if abs(change) >= cfg.vix_change_pct:
            events.append(
                _event(
                    TriggerType.VOLATILITY_EXPANSION_CONTRACTION,
                    current,
                    significance=_scale(change, cfg.vix_change_pct),
                    measure="india_vix",
                    direction="expansion" if change > 0 else "contraction",
                    change_pct=round(change, 3),
                    from_value=before_vix,
                    to_value=now_vix,
                )
            )

    before_rv = previous.volatility_state.realized_volatility
    now_rv = current.volatility_state.realized_volatility
    if before_rv is not None and now_rv is not None and before_rv > 0:
        change = (now_rv - before_rv) / before_rv * 100
        if abs(change) >= cfg.realized_vol_change_pct:
            events.append(
                _event(
                    TriggerType.VOLATILITY_EXPANSION_CONTRACTION,
                    current,
                    significance=_scale(change, cfg.realized_vol_change_pct),
                    measure="realized_volatility",
                    direction="expansion" if change > 0 else "contraction",
                    change_pct=round(change, 3),
                )
            )
    return events


def detect_iv_change(
    previous: MarketState | None, current: MarketState, cfg: TriggerEngineConfig
) -> list[Event]:
    if previous is None:
        return []
    before = previous.volatility_state.atm_iv
    now = current.volatility_state.atm_iv
    if before is None or now is None or before <= 0:
        return []
    change = (now - before) / before * 100
    if abs(change) < cfg.atm_iv_change_pct:
        return []
    return [
        _event(
            TriggerType.IV_EXPANSION_COLLAPSE,
            current,
            significance=_scale(change, cfg.atm_iv_change_pct),
            direction="expansion" if change > 0 else "collapse",
            change_pct=round(change, 3),
            from_iv=before,
            to_iv=now,
        )
    ]


# ------------------------------------------------------------------- options


def _chain_by_key(state: MarketState) -> dict[str, OptionQuote]:
    return {
        quote.contract.instrument_key: quote for quote in state.options_state.chain
    }


def detect_oi_change(
    previous: MarketState | None, current: MarketState, cfg: TriggerEngineConfig
) -> list[Event]:
    """Per-strike open interest builds and unwinds.

    Small strikes are excluded by `min_oi_for_change`: a strike going from ten
    lots to twenty has doubled and means nothing, and a ratio test alone would
    rank that above a 15% build on the ATM strike.
    """
    if previous is None:
        return []
    before_map = _chain_by_key(previous)
    events: list[Event] = []

    for quote in current.options_state.chain:
        before = before_map.get(quote.contract.instrument_key)
        if before is None:
            continue
        before_oi = before.open_interest
        if before_oi < cfg.min_oi_for_change:
            continue
        delta = quote.open_interest - before_oi
        ratio = delta / before_oi
        if abs(ratio) < cfg.oi_change_ratio:
            continue
        trigger = (
            TriggerType.LARGE_OI_ADDITION if delta > 0 else TriggerType.LARGE_OI_UNWINDING
        )
        events.append(
            _event(
                trigger,
                current,
                significance=_scale(ratio, cfg.oi_change_ratio),
                strike=float(quote.contract.strike),
                option_type=str(quote.contract.option_type),
                from_oi=before_oi,
                to_oi=quote.open_interest,
                ratio=round(ratio, 4),
            )
        )
    return events


def detect_oi_migration(
    previous: MarketState | None, current: MarketState, cfg: TriggerEngineConfig
) -> list[Event]:
    """The strike carrying the most open interest has moved.

    Worth its own trigger because it is where the market thinks the index will
    settle, and a shift in it is a shift in consensus rather than a change in
    magnitude.
    """
    if previous is None:
        return []
    events: list[Event] = []
    for option_type in (OptionType.CE, OptionType.PE):
        before = _max_oi_strike(previous, option_type)
        now = _max_oi_strike(current, option_type)
        if before is None or now is None or before == now:
            continue
        events.append(
            _event(
                TriggerType.OI_MIGRATION,
                current,
                significance=0.6,
                option_type=str(option_type),
                from_strike=float(before),
                to_strike=float(now),
                direction="up" if now > before else "down",
            )
        )
    return events


def _max_oi_strike(state: MarketState, option_type: OptionType) -> Decimal | None:
    legs = [
        quote
        for quote in state.options_state.chain
        if quote.contract.option_type is option_type and quote.open_interest > 0
    ]
    if not legs:
        return None
    return max(legs, key=lambda quote: quote.open_interest).contract.strike


def detect_premium_movement(
    previous: MarketState | None, current: MarketState, cfg: TriggerEngineConfig
) -> list[Event]:
    """A large move in the at-the-money premium.

    Priced from the mid rather than the last trade, for the same reason the
    adapter marks IV to the book: a stale LTP produces a phantom move.
    """
    if previous is None:
        return []
    spot = current.index_state.quote.ltp
    events: list[Event] = []
    for option_type in (OptionType.CE, OptionType.PE):
        before = _atm_leg(previous, option_type, previous.index_state.quote.ltp)
        now = _atm_leg(current, option_type, spot)
        if before is None or now is None:
            continue
        if before.contract.strike != now.contract.strike:
            # A different strike is now ATM, so this is not the same premium
            # moving. OI migration and price movement cover that case.
            continue
        before_mid = before.mid
        if before_mid <= 0:
            continue
        change = float((now.mid - before_mid) / before_mid * 100)
        if abs(change) < cfg.premium_move_pct:
            continue
        events.append(
            _event(
                TriggerType.LARGE_PREMIUM_MOVEMENT,
                current,
                significance=_scale(change, cfg.premium_move_pct),
                option_type=str(option_type),
                strike=float(now.contract.strike),
                change_pct=round(change, 3),
                from_price=float(before_mid),
                to_price=float(now.mid),
            )
        )
    return events


def _atm_leg(
    state: MarketState, option_type: OptionType, spot: Decimal
) -> OptionQuote | None:
    legs = [
        quote
        for quote in state.options_state.chain
        if quote.contract.option_type is option_type
    ]
    if not legs:
        return None
    return min(legs, key=lambda quote: abs(quote.contract.strike - spot))


def detect_gamma_concentration_change(
    previous: MarketState | None, current: MarketState, cfg: TriggerEngineConfig
) -> list[Event]:
    """Change in how concentrated gamma is around one strike.

    Depends on computed greeks, so it is silent for any leg without an IV —
    which is the honest outcome for a strike too wide to mark.
    """
    if previous is None:
        return []
    before = _gamma_concentration(previous)
    now = _gamma_concentration(current)
    if before is None or now is None:
        return []
    change = now - before
    if abs(change) < cfg.gamma_concentration_change:
        return []
    return [
        _event(
            TriggerType.GAMMA_CONCENTRATION_CHANGE,
            current,
            significance=_scale(change, cfg.gamma_concentration_change),
            from_concentration=round(before, 4),
            to_concentration=round(now, 4),
            direction="tightening" if change > 0 else "loosening",
        )
    ]


def _gamma_concentration(state: MarketState) -> float | None:
    """Share of total gamma sitting on the single largest strike."""
    per_strike: dict[Decimal, float] = {}
    for quote in state.options_state.chain:
        if quote.greeks is None:
            continue
        gamma = abs(float(quote.greeks.gamma))
        per_strike[quote.contract.strike] = per_strike.get(
            quote.contract.strike, 0.0
        ) + gamma
    total = sum(per_strike.values())
    if total <= 0:
        return None
    return max(per_strike.values()) / total


def detect_liquidity_deterioration(
    previous: MarketState | None, current: MarketState, cfg: TriggerEngineConfig
) -> list[Event]:
    """The chain's median relative spread has widened materially.

    Median rather than worst: one abandoned far-wing strike widening does not
    make the chain untradeable, and the worst-case measure would fire on it
    every session.
    """
    if previous is None:
        return []
    before = _median_spread(previous)
    now = _median_spread(current)
    if before is None or now is None:
        return []
    if before < cfg.min_spread_for_deterioration:
        # A spread going from 0.1% to 0.2% has doubled and is still excellent.
        return []
    ratio = now / before
    if ratio < cfg.spread_deterioration_ratio:
        return []
    return [
        _event(
            TriggerType.LIQUIDITY_DETERIORATION,
            current,
            significance=_scale(ratio, cfg.spread_deterioration_ratio),
            from_spread=round(before, 5),
            to_spread=round(now, 5),
            ratio=round(ratio, 3),
        )
    ]


def _median_spread(state: MarketState) -> float | None:
    spreads = sorted(
        float(quote.relative_spread)
        for quote in state.options_state.chain
        if quote.relative_spread is not None
    )
    if not spreads:
        return None
    middle = len(spreads) // 2
    if len(spreads) % 2:
        return spreads[middle]
    return (spreads[middle - 1] + spreads[middle]) / 2


# -------------------------------------------------------------- constituents


def detect_breadth_change(
    previous: MarketState | None, current: MarketState, cfg: TriggerEngineConfig
) -> list[Event]:
    """Requires a constituent provider, and stays silent without one."""
    if previous is None or previous.analysis is None or current.analysis is None:
        return []
    before = previous.analysis.constituents.breadth_score
    now = current.analysis.constituents.breadth_score
    change = now - before
    if abs(change) < cfg.breadth_change:
        return []
    return [
        _event(
            TriggerType.BREADTH_CHANGE,
            current,
            significance=_scale(change, cfg.breadth_change),
            from_breadth=round(before, 4),
            to_breadth=round(now, 4),
            direction="improving" if change > 0 else "deteriorating",
        )
    ]


def detect_major_constituent_movement(
    previous: MarketState | None, current: MarketState, cfg: TriggerEngineConfig
) -> list[Event]:
    """A heavyweight moving.

    Weight-gated on purpose: a 2% move in a 0.4%-weight name cannot shift the
    index, and waking a full analysis for it is the kind of noise that makes
    an event engine worse than a timer.
    """
    if previous is None:
        return []
    before_map = {quote.symbol: quote for quote in previous.constituent_state.quotes}
    weights = current.constituent_state.weights
    events: list[Event] = []

    for quote in current.constituent_state.quotes:
        weight = weights.get(quote.symbol)
        if weight is None or weight < cfg.min_constituent_weight:
            continue
        before = before_map.get(quote.symbol)
        if before is None or before.ltp <= 0:
            continue
        change = float((quote.ltp - before.ltp) / before.ltp * 100)
        if abs(change) < cfg.constituent_move_pct:
            continue
        events.append(
            _event(
                TriggerType.MAJOR_CONSTITUENT_MOVEMENT,
                current,
                significance=_scale(change, cfg.constituent_move_pct),
                symbol=quote.symbol,
                weight=weight,
                change_pct=round(change, 3),
            )
        )
    return events


def detect_sector_leadership_change(
    previous: MarketState | None, current: MarketState, cfg: TriggerEngineConfig
) -> list[Event]:
    if previous is None:
        return []
    before = previous.sector_state.sector_returns
    now = current.sector_state.sector_returns
    if not before or not now:
        return []
    before_leader = max(before, key=lambda sector: before[sector])
    now_leader = max(now, key=lambda sector: now[sector])
    if before_leader == now_leader:
        return []
    return [
        _event(
            TriggerType.SECTOR_LEADERSHIP_CHANGE,
            current,
            significance=0.5,
            from_leader=before_leader,
            to_leader=now_leader,
            leader_return=round(now[now_leader], 4),
        )
    ]


def detect_contribution_change(
    previous: MarketState | None, current: MarketState, cfg: TriggerEngineConfig
) -> list[Event]:
    if previous is None or previous.analysis is None or current.analysis is None:
        return []
    before = previous.analysis.constituents.concentration_score
    now = current.analysis.constituents.concentration_score
    change = now - before
    if abs(change) < cfg.contribution_change:
        return []
    return [
        _event(
            TriggerType.LARGE_CONTRIBUTION_CHANGE,
            current,
            significance=_scale(change, cfg.contribution_change),
            from_concentration=round(before, 4),
            to_concentration=round(now, 4),
        )
    ]


# ---------------------------------------------------------------------- time


_SESSION_TRIGGERS: dict[MarketSessionState, TriggerType] = {
    MarketSessionState.PRE_MARKET: TriggerType.PRE_MARKET,
    MarketSessionState.OPENING: TriggerType.MARKET_OPEN,
    MarketSessionState.CLOSING: TriggerType.PRE_CLOSE,
    MarketSessionState.CLOSED: TriggerType.END_OF_DAY,
}


def detect_session_transition(
    previous: MarketState | None, current: MarketState, cfg: TriggerEngineConfig
) -> list[Event]:
    """Session boundaries, from the snapshot's own timestamp.

    These are the one family that fires on the first tick: the session state is
    a fact about the clock the snapshot carries, not a change between two of
    them, and a process starting at 09:15 needs to know the market just opened.
    """
    trigger = _SESSION_TRIGGERS.get(current.session_state)
    if trigger is None:
        return []
    if previous is not None and previous.session_state is current.session_state:
        return []
    return [
        _event(
            trigger,
            current,
            significance=0.9,
            session_state=str(current.session_state),
            previous_session_state=(
                str(previous.session_state) if previous is not None else None
            ),
        )
    ]


def detect_opening_range_completion(
    previous: MarketState | None, current: MarketState, cfg: TriggerEngineConfig
) -> list[Event]:
    opening = current.index_state.opening_range
    if opening is None or not opening.completed:
        return []
    if previous is not None:
        before = previous.index_state.opening_range
        if before is not None and before.completed:
            return []
    return [
        _event(
            TriggerType.OPENING_RANGE_COMPLETION,
            current,
            significance=0.75,
            high=float(opening.high),
            low=float(opening.low),
        )
    ]


def detect_expiry_phase(
    previous: MarketState | None, current: MarketState, cfg: TriggerEngineConfig
) -> list[Event]:
    """Entering the final phase before expiry.

    Fires on the crossing, not on every tick inside it: pin risk and
    collapsing time value are a change of context, and re-announcing them
    every few seconds would drown everything else.
    """
    days = current.volatility_state.days_to_expiry
    if days is None or days > cfg.expiry_phase_days:
        return []
    if previous is not None:
        before = previous.volatility_state.days_to_expiry
        if before is not None and before <= cfg.expiry_phase_days:
            return []
    return [
        _event(
            TriggerType.EXPIRY_PHASE,
            current,
            significance=0.9,
            days_to_expiry=days,
            expiry=(
                current.options_state.expiry.isoformat()
                if current.options_state.expiry
                else None
            ),
        )
    ]


def make_heartbeat_detector(last_beat: dict[str, datetime]) -> Detector:
    """A periodic wake-up, so a quiet market is still looked at.

    Takes its cadence from snapshot timestamps rather than a timer, which is
    what lets a backtest produce the same heartbeats as a live session over
    the same data. The caller owns the `last_beat` dict so the engine stays a
    pure function of its inputs plus explicit state.
    """

    def detect(
        previous: MarketState | None, current: MarketState, cfg: TriggerEngineConfig
    ) -> list[Event]:
        if current.session_state not in (
            MarketSessionState.ACTIVE,
            MarketSessionState.OPENING,
            MarketSessionState.CLOSING,
        ):
            # A heartbeat outside the session would wake the pipeline all
            # night to re-analyse a closed market.
            return []
        previous_beat = last_beat.get("at")
        if previous_beat is not None:
            elapsed = (current.timestamp - previous_beat).total_seconds()
            if elapsed < cfg.heartbeat_seconds:
                return []
        last_beat["at"] = current.timestamp
        return [
            _event(
                TriggerType.PERIODIC_HEARTBEAT,
                current,
                significance=0.4,
                session_state=str(current.session_state),
            )
        ]

    return detect


# ------------------------------------------------------------------ registry

MARKET_DETECTORS: tuple[Detector, ...] = (
    detect_price_movement,
    detect_exceptional_move,
    detect_breakout,
    detect_opening_range_event,
    detect_vwap_crossing,
    detect_level_test,
    detect_volume_anomaly,
    detect_volatility_change,
)

OPTION_DETECTORS: tuple[Detector, ...] = (
    detect_iv_change,
    detect_oi_change,
    detect_oi_migration,
    detect_premium_movement,
    detect_gamma_concentration_change,
    detect_liquidity_deterioration,
)

CONSTITUENT_DETECTORS: tuple[Detector, ...] = (
    detect_breadth_change,
    detect_major_constituent_movement,
    detect_sector_leadership_change,
    detect_contribution_change,
)

TIME_DETECTORS: tuple[Detector, ...] = (
    detect_session_transition,
    detect_opening_range_completion,
    detect_expiry_phase,
)

# The trigger types no detector here can produce, and why. Kept as data so
# the gap is checkable rather than a matter of reading the code: a scheduled
# economic event, an RBI decision, the Union Budget and an index rebalance are
# all calendar facts, and no free Indian source for that calendar was found.
# `ScheduledEventCalendar` is where they enter the system when one exists.
CALENDAR_ONLY_TRIGGERS: frozenset[TriggerType] = frozenset(
    {
        TriggerType.MAJOR_SCHEDULED_ECONOMIC_EVENT,
        TriggerType.RBI_EVENT,
        TriggerType.BUDGET_EVENT_RISK,
        TriggerType.INDEX_REBALANCE,
    }
)


def _atr(state: MarketState) -> Decimal | None:
    if state.analysis is None:
        return None
    return state.analysis.index.atr
