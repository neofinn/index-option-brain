"""Spec §22/§26. Replay the deterministic decision chain over real history.

The same brain code runs here as runs live — this module swaps the data
source and nothing else. There is no backtest-specific strategy logic, which
is the whole point of the spec's insistence that RunMode change only the
edges of the system.

Why the tradeable decision is not what gets measured
----------------------------------------------------
The first version of this module recorded `signal.direction` and reported
that the engine was neutral on 239 of 239 sessions. That number was about
this harness, not about the strategy. With no historical chain the Scenario
Engine's NO_TRADE case scores ~0.62 on the evidence "option chain is only 0%
complete" and "chain liquidity 0.00 would surrender any edge" — correct
reasoning, since options cannot be traded without a chain — and it
out-scores a BEARISH_CONTINUATION sitting at 0.438 on a -0.93 index
composite. The veto is right and it is total, so `signal.direction` in a
replay measures the missing data and nothing else.

So the thing measured here is the **best directional scenario** the analysis
produced, whether or not a trade could have been placed on it. That is a
real and separable question: does the analysis layer's directional read lead
price? A yes is necessary for the system to be worth running and is not
sufficient. A no means the rest cannot rescue it.

What this can honestly evaluate, and what it cannot
---------------------------------------------------
NSE publishes daily index bars in its end-of-day archive, so the **Index
brain, Regime Engine, Scenario Engine, Signal Engine and Strategy Engine can
all be replayed on real data.** Everything up to the point where a structure
has been chosen is measurable.

It publishes **no historical option chains**, and neither does any free
source. So the Strike Engine, the cost model, position sizing and P&L are
*not* replayable: a strike cannot be selected from a chain that does not
exist, and inventing one would produce a backtest whose returns are a
property of the invention. This module therefore stops where the data stops
and reports **signal quality**, not profit.

That is a real limitation, not a soft one. A signal that leads price is
necessary for the system to work and is not sufficient — a correct direction
can still lose money to spread, theta and a bad strike. Read the output as
"is the decision layer informative", never as "this is what it would have
made".

No lookahead
------------
The one property that decides whether any of this means anything. At each
step the state is built from bars `[0 .. i]` only, and the quote is bar `i`
— the session that has just closed. Forward returns are computed from bars
strictly after `i`, and are never visible to the brain. `test_replay.py`
asserts this by feeding a series whose future is a step change and checking
the signal does not move before it.

Because the daily archive is one bar per session, a replayed cycle is one
decision per day, taken on the close. That is coarser than the live loop's
20 seconds, and it is the honest resolution of the available data.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from index_option_brain.brain.pipeline import QuantitativeBrain
from index_option_brain.contracts.analysis import RegimeState
from index_option_brain.contracts.enums import Direction, MarketSessionState, StrategyType
from index_option_brain.contracts.instruments import Bar, IndexQuote, IndexSpec
from index_option_brain.contracts.market_state import (
    ConstituentState,
    IndexState,
    MarketState,
    OptionsState,
    SectorState,
    VolatilityState,
)
from index_option_brain.contracts.signal import Signal

#: Sessions of history required before the first decision is taken. The Index
#: brain scales its confidence by len(daily)/min_daily_bars (30), so replaying
#: from bar 1 would spend the early window measuring the warm-up rather than
#: the market.
DEFAULT_WARMUP = 30


@dataclass(frozen=True)
class ReplayCycle:
    """One session's decision, and what the market did next.

    `forward_returns` is keyed by horizon in sessions. A horizon missing from
    the mapping ran off the end of the data — absent, never zero.
    """

    index_symbol: str
    as_of: datetime
    close: Decimal
    bars_seen: int
    regime: RegimeState
    signal: Signal
    strategy: StrategyType
    forward_returns: dict[int, float]
    view: Direction = Direction.NEUTRAL
    """Direction of the best-scoring directional scenario, neutral if none.

    Not `signal.direction`: see the module docstring. In a replay the
    tradeable signal is vetoed by the absent chain on every session, so it
    carries no information about the analysis.
    """
    view_score: float = 0.0
    view_kind: str = ""

    @property
    def direction(self) -> Direction:
        """The analysis layer's directional read — what this harness scores."""
        return self.view

    @property
    def tradeable_direction(self) -> Direction:
        """What the Signal Engine would actually have acted on."""
        return self.signal.direction

    @property
    def took_a_view(self) -> bool:
        return self.view is not Direction.NEUTRAL


def state_from_bars(
    *,
    index_symbol: str,
    bars: Sequence[Bar],
    spec: IndexSpec | None = None,
    vix_bars: Sequence[Bar] | None = None,
    vix_year_range: tuple[float, float] | None = None,
) -> MarketState:
    """A MarketState describing the close of `bars[-1]`.

    `daily_bars` excludes the final bar and `quote` carries it, matching the
    live contract exactly: IndexState.daily_bars holds *completed prior*
    sessions, and appending the session being decided on would let every
    previous-day level silently include today.

    Options state is left empty: NSE publishes no historical chains, so open
    interest, walls, skew and the forward are genuinely unmeasured here and
    the Options brain reports that absence — the same code path it takes live
    when a chain cannot be read.

    Volatility state is populated when `vix_bars` is supplied, because India
    VIX *is* in the daily archive. That matters more than it sounds: the
    Signal Engine requires two domains to express a view before it credits
    conviction (`min_participating_domains`), so a replay on index bars alone
    can never produce a directional signal regardless of what the market did.
    A replay without VIX measures the warm-up, not the strategy.
    """
    if len(bars) < 2:
        raise ValueError("A replayed state needs at least two bars: one prior, one current")

    current = bars[-1]
    prior = list(bars[:-1])
    quote = IndexQuote(
        symbol=index_symbol,
        timestamp=current.timestamp,
        # The decision is taken on the close, so the close *is* the last price.
        ltp=current.close,
        open=current.open,
        high=current.high,
        low=current.low,
        previous_close=prior[-1].close,
    )
    return MarketState(
        timestamp=current.timestamp,
        # A daily bar describes a whole session; by the time it exists the
        # session is over.
        session_state=MarketSessionState.CLOSED,
        index_state=IndexState(quote=quote, spec=spec, daily_bars=prior),
        constituent_state=ConstituentState(),
        sector_state=SectorState(),
        options_state=OptionsState(),
        volatility_state=_volatility_state(vix_bars, vix_year_range),
    )


def _volatility_state(
    vix_bars: Sequence[Bar] | None,
    year_range: tuple[float, float] | None,
) -> VolatilityState:
    """Volatility as of the same close, from the archive's India VIX row.

    `atm_iv` stays None: VIX is a 30-day interpolated index, not the ATM
    implied volatility of the weekly the system trades. Passing it off as
    `atm_iv` would let every richness comparison downstream read a
    same-named but different measurement — the exact substitution this
    codebase refuses elsewhere. It is supplied as India VIX, which is what
    it is, and the Volatility brain already knows how to rank that against
    its 52-week range.
    """
    if not vix_bars or len(vix_bars) < 2:
        return VolatilityState()
    closes = [float(b.close) for b in vix_bars]
    high, low = (year_range if year_range else (max(closes), min(closes)))
    return VolatilityState(
        india_vix=closes[-1],
        india_vix_previous_close=closes[-2],
        india_vix_year_high=high,
        india_vix_year_low=low,
    )


class DailyReplayEngine:
    """Walks a daily series forward, one decision per session.

    Holds a `QuantitativeBrain` and calls it exactly as the live engine does.
    No account or portfolio is supplied, so the Risk Engine does not run and
    no cycle here can present itself as authorized — authorizing a size
    against an account that never existed would be the worst kind of
    backtest.
    """

    def __init__(
        self,
        brain: QuantitativeBrain | None = None,
        *,
        warmup: int = DEFAULT_WARMUP,
        horizons: Sequence[int] = (1, 3, 5),
        min_view_score: float = 0.40,
    ) -> None:
        """
        `min_view_score` is the scenario score below which a directional read
        is not counted as a view. Every scenario the engine generates carries
        some score, so without a floor this would evaluate noise. 0.40 is
        deliberately close to the Signal Engine's own `min_score` of 0.35 —
        the point is to measure the reads the engine would have acted on had
        a chain existed, not every read it entertained.
        """
        self._brain = brain or QuantitativeBrain()
        self._warmup = max(2, warmup)
        self._horizons = tuple(sorted(set(horizons)))
        self._min_view_score = min_view_score

    def run(
        self,
        index_symbol: str,
        bars: Sequence[Bar],
        *,
        spec: IndexSpec | None = None,
        vix_bars: Sequence[Bar] | None = None,
    ) -> list[ReplayCycle]:
        """Replay `bars`, optionally with an aligned India VIX series.

        `vix_bars` must be the same length as `bars` and cover the same
        sessions — `NseArchiveAdapter.get_many_index_bars` returns series
        aligned by construction. A misaligned series is refused rather than
        silently zipped, because an off-by-one there reads yesterday's
        volatility into today's decision and would look like signal.
        """
        if vix_bars is not None and len(vix_bars) != len(bars):
            raise ValueError(
                f"VIX series has {len(vix_bars)} bars against {len(bars)} index "
                "bars; a replay cannot align them and will not guess"
            )
        vix_range: tuple[float, float] | None = None
        if vix_bars:
            closes = [float(b.close) for b in vix_bars]
            vix_range = (max(closes), min(closes))

        cycles: list[ReplayCycle] = []
        for index in range(self._warmup, len(bars)):
            # The slice is the whole guarantee: the brain is handed the past
            # and the present, never the future. The VIX slice is cut at the
            # same index for the same reason.
            visible = bars[: index + 1]
            state = state_from_bars(
                index_symbol=index_symbol,
                bars=visible,
                spec=spec,
                vix_bars=vix_bars[: index + 1] if vix_bars else None,
                vix_year_range=vix_range,
            )
            result = self._brain.run(state)
            if result.regime is None:
                continue
            directional = [
                s for s in result.scenarios if s.direction is not Direction.NEUTRAL
            ]
            leader = max(directional, key=lambda s: s.score, default=None)
            takes_view = leader is not None and leader.score >= self._min_view_score
            view = leader.direction if (leader and takes_view) else Direction.NEUTRAL
            view_kind = str(leader.kind) if (leader and takes_view) else ""
            cycles.append(
                ReplayCycle(
                    index_symbol=index_symbol,
                    as_of=state.timestamp,
                    close=bars[index].close,
                    bars_seen=len(visible),
                    regime=result.regime,
                    signal=result.signal,
                    strategy=result.selected_strategy,
                    forward_returns=self._forward(bars, index),
                    view=view,
                    view_score=leader.score if leader is not None else 0.0,
                    view_kind=view_kind,
                )
            )
        return cycles

    def _forward(self, bars: Sequence[Bar], index: int) -> dict[int, float]:
        """Close-to-close percentage change over each horizon.

        Computed from bars the brain was not shown, and omitted entirely
        where the horizon runs past the end of the series — so a truncated
        tail cannot be read as a run of flat outcomes.
        """
        base = float(bars[index].close)
        if base <= 0:
            return {}
        out: dict[int, float] = {}
        for horizon in self._horizons:
            target = index + horizon
            if target >= len(bars):
                continue
            out[horizon] = (float(bars[target].close) - base) / base * 100.0
        return out


@dataclass(frozen=True)
class DirectionStats:
    """How a group of decisions actually fared at one horizon.

    Carries its own standard error because a hit rate without one is an
    invitation to over-read it. The first useful run of this harness produced
    a 64.7% bullish hit rate against a 50.7% base rate — a 14-point edge that
    sounds decisive and sits inside one standard error at n=17.
    """

    label: str
    count: int
    mean_return: float | None
    median_return: float | None
    hit_rate: float | None
    """Share that moved in the signalled direction. None for a neutral group,
    where "correct" has no meaning."""

    @property
    def is_measurable(self) -> bool:
        return self.count > 0 and self.mean_return is not None

    @property
    def hit_rate_standard_error(self) -> float | None:
        """Standard error of the hit rate, as a fraction.

        The binomial se, sqrt(p(1-p)/n). None where there is no hit rate to
        put an error on.
        """
        if self.hit_rate is None or self.count == 0:
            return None
        p = self.hit_rate
        return float((p * (1.0 - p) / self.count) ** 0.5)

    def beats(self, base_rate: float | None, *, sigmas: float = 2.0) -> bool | None:
        """Whether the hit rate clears `base_rate` by `sigmas` standard errors.

        None when either side is unmeasured. Two sigmas is the default
        because one is not evidence — and on the samples this harness can
        currently reach, almost nothing clears two.
        """
        se = self.hit_rate_standard_error
        if se is None or base_rate is None or self.hit_rate is None:
            return None
        if se == 0.0:
            return self.hit_rate > base_rate
        return (self.hit_rate - base_rate) > sigmas * se


@dataclass(frozen=True)
class ReplayReport:
    """The result of a replay, including the base rate to judge it against.

    `base_rate_up` is the unconditional share of sessions that rose. Without
    it a hit rate is meaningless: 55% correct on a tape that rose 55% of the
    time is not skill, it is the tape. Every comparison this report supports
    is against that number, never against 50%.
    """

    index_symbol: str
    horizon: int
    sessions: int
    first: datetime | None
    last: datetime | None
    base_rate_up: float | None
    mean_session_return: float | None
    by_direction: dict[str, DirectionStats]
    by_regime: dict[str, int]
    by_strategy: dict[str, int]
    tradeable_views: int = 0
    """Sessions the Signal Engine itself would have acted on.

    Zero throughout a chainless replay, and reported so the difference
    between "the analysis had no view" and "the analysis had a view the
    missing chain vetoed" stays visible in the output.
    """

    @property
    def no_view_share(self) -> float | None:
        """Share of sessions the engine declined to take a view on.

        Reported prominently because a system that is silent 95% of the time
        is a different proposition from one that is silent 20% of the time,
        and both are legitimate.
        """
        if self.sessions == 0:
            return None
        neutral = self.by_direction.get(str(Direction.NEUTRAL))
        return (neutral.count / self.sessions) if neutral else 0.0

    def edge_over_base_rate(self) -> float | None:
        """Bullish-signal hit rate minus the base rate, in percentage points.

        None when either side is unmeasured. A positive number is a necessary
        condition for the decision layer to be worth anything — not a
        sufficient one, since it says nothing about spread, theta or strike
        selection, and not a significant one unless `edge_is_significant`
        agrees.
        """
        bullish = self.by_direction.get(str(Direction.BULLISH))
        if bullish is None or bullish.hit_rate is None or self.base_rate_up is None:
            return None
        return (bullish.hit_rate - self.base_rate_up) * 100.0

    def edge_is_significant(self, *, sigmas: float = 2.0) -> bool | None:
        """Whether the bullish edge survives its own error bar."""
        bullish = self.by_direction.get(str(Direction.BULLISH))
        if bullish is None:
            return None
        return bullish.beats(self.base_rate_up, sigmas=sigmas)

    @property
    def smallest_directional_sample(self) -> int:
        """The smaller of the two directional counts.

        The number that governs how much of this report can be believed. A
        report is not worth reading below ~30 either side, and this harness
        reaches that only over multiple years.
        """
        return min(
            self.by_direction[str(Direction.BULLISH)].count,
            self.by_direction[str(Direction.BEARISH)].count,
        )


def _stats(label: str, returns: list[float], *, expect_up: bool | None) -> DirectionStats:
    if not returns:
        return DirectionStats(label=label, count=0, mean_return=None,
                              median_return=None, hit_rate=None)
    hit: float | None = None
    if expect_up is not None:
        wins = sum(1 for r in returns if (r > 0) == expect_up)
        hit = wins / len(returns)
    return DirectionStats(
        label=label,
        count=len(returns),
        mean_return=statistics.fmean(returns),
        median_return=statistics.median(returns),
        hit_rate=hit,
    )


def evaluate(
    cycles: Sequence[ReplayCycle],
    *,
    horizon: int = 1,
    all_bars: Sequence[Bar] | None = None,
) -> ReplayReport:
    """Summarise a replay at one horizon, against the unconditional base rate.

    `all_bars` supplies the base rate over the *whole* series when given.
    Falling back to the replayed window is acceptable but weaker — the window
    is already conditioned on the warm-up having passed.
    """
    scored = [c for c in cycles if horizon in c.forward_returns]

    buckets: dict[str, list[float]] = {}
    for cycle in scored:
        buckets.setdefault(str(cycle.direction), []).append(
            cycle.forward_returns[horizon]
        )

    by_direction = {
        str(Direction.BULLISH): _stats(str(Direction.BULLISH),
                                       buckets.get(str(Direction.BULLISH), []),
                                       expect_up=True),
        str(Direction.BEARISH): _stats(str(Direction.BEARISH),
                                       buckets.get(str(Direction.BEARISH), []),
                                       expect_up=False),
        str(Direction.NEUTRAL): _stats(str(Direction.NEUTRAL),
                                       buckets.get(str(Direction.NEUTRAL), []),
                                       expect_up=None),
    }

    base_source: list[float] = []
    if all_bars is not None and len(all_bars) > horizon:
        base_source = [
            (float(all_bars[i + horizon].close) - float(all_bars[i].close))
            / float(all_bars[i].close)
            * 100.0
            for i in range(len(all_bars) - horizon)
            if float(all_bars[i].close) > 0
        ]
    elif scored:
        base_source = [c.forward_returns[horizon] for c in scored]

    regimes: dict[str, int] = {}
    strategies: dict[str, int] = {}
    for cycle in cycles:
        key = str(cycle.regime.regime)
        regimes[key] = regimes.get(key, 0) + 1
        skey = str(cycle.strategy)
        strategies[skey] = strategies.get(skey, 0) + 1

    return ReplayReport(
        tradeable_views=sum(
            1 for c in cycles if c.tradeable_direction is not Direction.NEUTRAL
        ),
        index_symbol=cycles[0].index_symbol if cycles else "",
        horizon=horizon,
        sessions=len(scored),
        first=cycles[0].as_of if cycles else None,
        last=cycles[-1].as_of if cycles else None,
        base_rate_up=(
            sum(1 for r in base_source if r > 0) / len(base_source)
            if base_source
            else None
        ),
        mean_session_return=statistics.fmean(base_source) if base_source else None,
        by_direction=by_direction,
        by_regime=regimes,
        by_strategy=strategies,
    )


def bar_at(day: datetime, *, open_: float, high: float, low: float, close: float) -> Bar:
    """Small helper for constructing a replay series in tests and scripts."""
    return Bar(
        timestamp=day if day.tzinfo else day.replace(tzinfo=UTC),
        open=Decimal(str(open_)),
        high=Decimal(str(high)),
        low=Decimal(str(low)),
        close=Decimal(str(close)),
    )
