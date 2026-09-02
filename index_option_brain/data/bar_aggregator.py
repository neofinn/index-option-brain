"""Build OHLCV bars from a stream of live index snapshots.

Why this exists
---------------
NSE's public feed serves a live index snapshot but no history: the historical
endpoint answers an automated client with an anti-bot page. The Index brain
needs bars — ATR, RSI, swing structure and previous-session levels all come
from them — so with no history endpoint there are exactly two honest options:
take bars from a broker adapter that has them, or observe the market and build
them going forward. This is the second.

What it will and will not do
----------------------------
It records what it saw. It does not interpolate a bucket it has no
observations for, it does not extend the last bar to fill a gap, and it does
not keep a bucket whose opening print it missed. A bar built from the second
half of a five-minute window, labelled with that window's start, is a candle
the market never printed — and every level derived from it (opening range,
swing pivots, ATR) would inherit the error silently.

Three details that are easy to get wrong
----------------------------------------
* **Buckets are keyed by the snapshot's own timestamp, never by wall clock.**
  This is what makes holidays and weekends free: NSE keeps serving the last
  session's data with the last session's timestamp, so a Sunday poll reports a
  timestamp the aggregator has already seen and is discarded as non-advancing.
  Using `now()` would manufacture a week of flat candles over a long weekend.

* **Outside session hours, intraday snapshots are ignored.** The endpoint
  answers all night with the closing price. Aggregating that would produce
  hundreds of flat bars and a completely fictitious ATR.

* **Daily bars are not aggregated from ticks.** The snapshot already carries
  the exchange's own session open, high and low, which cover the whole session
  including the part before this process started watching. Observed extremes
  would only cover the window we were connected for, so for `BarInterval.DAY`
  the exchange's figures are used directly and only the close is tracked.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta, timezone
from decimal import Decimal

from index_option_brain.contracts.enums import BarInterval
from index_option_brain.contracts.instruments import Bar, IndexQuote, IndexSpec
from index_option_brain.data.adapters.base import DataAdapterError, IndexDataAdapter

IST = timezone(timedelta(hours=5, minutes=30), name="IST")

SESSION_OPEN_IST = time(9, 15)
SESSION_CLOSE_IST = time(15, 30)

_INTERVAL_SECONDS: dict[BarInterval, int] = {
    BarInterval.MINUTE_1: 60,
    BarInterval.MINUTE_5: 300,
    BarInterval.MINUTE_15: 900,
}

# A bucket is kept only if its first observation arrived within this fraction
# of the interval. A five-minute bar whose first print lands two minutes in has
# no opening price, and inventing one from a later print misstates the candle
# in exactly the direction the market moved.
DEFAULT_OPEN_LAG_FRACTION = 0.25


def interval_seconds(interval: BarInterval) -> int:
    try:
        return _INTERVAL_SECONDS[interval]
    except KeyError:
        raise DataAdapterError(
            f"{interval} is not an intraday interval that can be aggregated"
        ) from None


def session_open(moment: datetime) -> datetime:
    """The UTC instant the session containing `moment` opened."""
    local = moment.astimezone(IST)
    return datetime.combine(local.date(), SESSION_OPEN_IST, tzinfo=IST).astimezone(
        UTC
    )


def session_close(moment: datetime) -> datetime:
    local = moment.astimezone(IST)
    return datetime.combine(local.date(), SESSION_CLOSE_IST, tzinfo=IST).astimezone(
        UTC
    )


def in_session(moment: datetime) -> bool:
    """Whether `moment` falls inside the Indian equity session.

    Weekends are excluded, but exchange holidays are not knowable from a
    calendar — a holiday shows up instead as a snapshot whose timestamp does
    not advance, which the aggregator discards anyway.
    """
    local = moment.astimezone(IST)
    if local.weekday() >= 5:
        return False
    return SESSION_OPEN_IST <= local.time() <= SESSION_CLOSE_IST


def is_bucket_start(moment: datetime) -> bool:
    """Whether a bar can begin at `moment`.

    Stricter than `in_session`: the close itself is inside the session but
    cannot open a bar, because there is no window after it. Without this
    distinction every overnight break would be counted as one missing bucket
    and a multi-day series would look permanently gappy.
    """
    return in_session(moment) and moment < session_close(moment)


def bucket_start(moment: datetime, interval: BarInterval) -> datetime:
    """The start of the bar `moment` belongs to.

    Aligned to the session open rather than to the hour. It happens that
    09:15 IST is a whole multiple of 1, 5 and 15 minutes past midnight UTC, so
    the two agree today — but anchoring to the session means a change to the
    opening time cannot silently shift every bucket boundary by a few minutes.
    """
    opened = session_open(moment)
    elapsed = int((moment - opened).total_seconds())
    step = interval_seconds(interval)
    return opened + timedelta(seconds=(elapsed // step) * step)


@dataclass
class _Forming:
    """The bar currently being built. Never handed out as completed."""

    start: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    complete_open: bool
    """False when the opening print was missed, which disqualifies the bar."""

    def update(self, price: Decimal) -> None:
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price

    def to_bar(self) -> Bar:
        return Bar(
            timestamp=self.start,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            # NSE publishes no index volume on the snapshot endpoint. Zero here
            # means "not measured" and no indicator in the system reads index
            # volume; a fabricated figure would be worse than an absent one.
            volume=0,
        )


@dataclass
class AggregationStats:
    """What the aggregator has and has not seen.

    Consumers need this to tell a short history from a gappy one: indicators
    read a bar list as a contiguous series, so a hole in it silently changes
    what ATR and RSI mean.
    """

    observations: int = 0
    stale_snapshots: int = 0
    """Snapshots whose timestamp did not advance — repeats, weekends, holidays."""
    out_of_session: int = 0
    bars_completed: int = 0
    bars_discarded_partial: int = 0
    """Buckets dropped because their opening print was missed."""
    missing_buckets: int = 0
    """Buckets that elapsed with no observation at all — a feed outage."""
    first_observation: datetime | None = None
    last_observation: datetime | None = None
    seeded_bars: int = 0

    @property
    def has_gaps(self) -> bool:
        return self.missing_buckets > 0 or self.bars_discarded_partial > 0


class LiveBarAggregator:
    """Accumulates one symbol's snapshots into bars of one interval."""

    def __init__(
        self,
        interval: BarInterval,
        *,
        max_bars: int = 500,
        open_lag_fraction: float = DEFAULT_OPEN_LAG_FRACTION,
    ) -> None:
        if max_bars < 1:
            raise ValueError("max_bars must be at least 1")
        self.interval = interval
        self._max_bars = max_bars
        self._is_daily = interval is BarInterval.DAY
        self._step = None if self._is_daily else interval_seconds(interval)
        self._open_lag = (
            None
            if self._step is None
            else timedelta(seconds=self._step * open_lag_fraction)
        )
        self._bars: list[Bar] = []
        self._forming: _Forming | None = None
        self._last_timestamp: datetime | None = None
        self.stats = AggregationStats()

    # ----------------------------------------------------------------- state

    @property
    def completed(self) -> list[Bar]:
        """Completed bars, oldest first, never including the forming one.

        The adapter contract is explicit that a partial candle must not appear
        here: the brains read the last daily bar as the previous session, and a
        forming bar in that position would corrupt every level derived from it.
        """
        return list(self._bars)

    @property
    def forming(self) -> Bar | None:
        """The in-progress bar, for display only.

        Exposed because an operations console showing a live chart needs it,
        and clearly separated so it cannot reach an indicator by accident.
        """
        return self._forming.to_bar() if self._forming is not None else None

    def seed(self, bars: list[Bar]) -> None:
        """Prime the series with history from a provider that has it.

        Seeded bars are trusted as given and are not re-derived. The point of
        seeding is a warm start: a broker adapter supplies the last few hundred
        candles, and this aggregator extends them live, so a fresh process is
        not blind for the first hour.
        """
        ordered = sorted(bars, key=lambda bar: bar.timestamp)
        self._bars = ordered[-self._max_bars :]
        self.stats.seeded_bars = len(self._bars)
        if self._bars:
            self._last_timestamp = self._bars[-1].timestamp

    # ----------------------------------------------------------- observation

    def observe(self, quote: IndexQuote) -> Bar | None:
        """Fold one snapshot in; return a bar if this observation completed one.

        Returns `None` most of the time — a completed bar is the exception, not
        the rule, and a caller must not treat `None` as a failure.
        """
        moment = quote.timestamp
        self.stats.observations += 1

        # Non-advancing timestamps are repeats of a snapshot already folded in.
        # This is also what makes weekends and exchange holidays inert.
        if self._last_timestamp is not None and moment <= self._last_timestamp:
            self.stats.stale_snapshots += 1
            return None

        if self._is_daily:
            completed = self._observe_daily(quote)
        else:
            completed = self._observe_intraday(quote)

        self._last_timestamp = moment
        if self.stats.first_observation is None:
            self.stats.first_observation = moment
        self.stats.last_observation = moment
        return completed

    def _observe_intraday(self, quote: IndexQuote) -> Bar | None:
        moment = quote.timestamp
        if not in_session(moment):
            # The endpoint answers all night with the closing price. Folding
            # that in would produce flat candles the market never printed.
            self.stats.out_of_session += 1
            return None

        start = bucket_start(moment, self.interval)
        forming = self._forming
        completed: Bar | None = None

        if forming is not None and start > forming.start:
            completed = self._close_bucket(forming, next_start=start)
            forming = None

        if forming is None:
            assert self._open_lag is not None
            forming = _Forming(
                start=start,
                open=quote.ltp,
                high=quote.ltp,
                low=quote.ltp,
                close=quote.ltp,
                complete_open=(moment - start) <= self._open_lag,
            )
            self._forming = forming
        else:
            forming.update(quote.ltp)

        return completed

    def _close_bucket(self, forming: _Forming, *, next_start: datetime) -> Bar | None:
        """Finish a bucket and count any buckets skipped between the two."""
        assert self._step is not None
        elapsed = int((next_start - forming.start).total_seconds())
        skipped = elapsed // self._step - 1
        if skipped > 0:
            # Only count buckets inside the session; an overnight break is not
            # a feed outage.
            self.stats.missing_buckets += self._session_buckets_between(
                forming.start, next_start
            )

        if not forming.complete_open:
            self.stats.bars_discarded_partial += 1
            return None

        bar = forming.to_bar()
        self._append(bar)
        self.stats.bars_completed += 1
        return bar

    def _session_buckets_between(self, start: datetime, end: datetime) -> int:
        """Count in-session buckets strictly between two bucket starts."""
        assert self._step is not None
        step = timedelta(seconds=self._step)
        moment = start + step
        missing = 0
        while moment < end:
            if is_bucket_start(moment):
                missing += 1
            moment += step
        return missing

    def _observe_daily(self, quote: IndexQuote) -> Bar | None:
        """Track the session bar from the exchange's own open, high and low.

        A pre-open snapshot is ignored: before 09:15 the endpoint reports the
        previous session's figures, and folding them into today's bar would
        duplicate yesterday.
        """
        moment = quote.timestamp
        local = moment.astimezone(IST)
        if local.weekday() >= 5 or local.time() < SESSION_OPEN_IST:
            self.stats.out_of_session += 1
            return None

        start = session_open(moment)
        forming = self._forming
        completed: Bar | None = None

        if forming is not None and start > forming.start:
            bar = forming.to_bar()
            self._append(bar)
            self.stats.bars_completed += 1
            completed = bar
            forming = None

        if forming is None:
            self._forming = _Forming(
                start=start,
                open=quote.open,
                high=quote.high,
                low=quote.low,
                close=quote.ltp,
                complete_open=True,
            )
        else:
            # The exchange's high and low are cumulative over the session, so
            # they are taken as given — but guarded with max/min so a glitched
            # snapshot reporting a lower high cannot shrink the bar.
            forming.high = max(forming.high, quote.high)
            forming.low = min(forming.low, quote.low)
            forming.close = quote.ltp

        return completed

    def _append(self, bar: Bar) -> None:
        self._bars.append(bar)
        if len(self._bars) > self._max_bars:
            del self._bars[: len(self._bars) - self._max_bars]

    def close_session(self) -> Bar | None:
        """Complete the bar in progress, for use at the session close.

        Called explicitly rather than inferred from a clock, so the same code
        path serves live, replay and backtest: nothing here reads `now()`.
        """
        forming = self._forming
        self._forming = None
        if forming is None:
            return None
        if not forming.complete_open:
            self.stats.bars_discarded_partial += 1
            return None
        bar = forming.to_bar()
        self._append(bar)
        self.stats.bars_completed += 1
        return bar


class AggregatingIndexAdapter(IndexDataAdapter):
    """Wrap an index adapter that has no history and give it bars over time.

    This is the composition the capability-split interfaces were designed for:
    NSE serves a live snapshot and no bars, so it is wrapped rather than
    extended, and every snapshot read for analysis doubles as a bar
    observation. Wrap a broker adapter's bars around it later and the seeding
    path replaces the cold start without anything downstream changing.

    Bars are only as long as this process has been watching, and `stats`
    reports exactly how long. A caller must not read a short series as a calm
    market.
    """

    def __init__(
        self,
        source: IndexDataAdapter,
        *,
        intervals: tuple[BarInterval, ...] = (BarInterval.MINUTE_5, BarInterval.DAY),
        max_bars: int = 500,
    ) -> None:
        self._source = source
        self._aggregators: dict[str, dict[BarInterval, LiveBarAggregator]] = {}
        self._intervals = intervals
        self._max_bars = max_bars

    def aggregator(self, symbol: str, interval: BarInterval) -> LiveBarAggregator:
        symbol = symbol.upper()
        if interval not in self._intervals:
            raise DataAdapterError(
                f"{interval} is not being aggregated for {symbol}. Tracked: "
                f"{[str(i) for i in self._intervals]}"
            )
        per_symbol = self._aggregators.setdefault(symbol, {})
        if interval not in per_symbol:
            per_symbol[interval] = LiveBarAggregator(
                interval, max_bars=self._max_bars
            )
        return per_symbol[interval]

    def seed(self, symbol: str, interval: BarInterval, bars: list[Bar]) -> None:
        self.aggregator(symbol, interval).seed(bars)

    def stats(self, symbol: str, interval: BarInterval) -> AggregationStats:
        return self.aggregator(symbol, interval).stats

    async def get_index_spec(self, symbol: str) -> IndexSpec:
        return await self._source.get_index_spec(symbol)

    async def get_index_quote(self, symbol: str) -> IndexQuote:
        """Fetch the live snapshot and fold it into every tracked interval."""
        quote = await self._source.get_index_quote(symbol)
        for interval in self._intervals:
            self.aggregator(symbol, interval).observe(quote)
        return quote

    async def get_index_bars(
        self, symbol: str, interval: BarInterval, count: int
    ) -> list[Bar]:
        """Whatever has been observed so far, newest `count` bars, oldest first.

        Returns fewer than requested — including none at all on a cold start —
        rather than raising or padding. The indicators are built to return
        `None` on insufficient data, so a short series degrades the analysis
        honestly, while padded bars would produce a confident wrong answer.
        """
        bars = self.aggregator(symbol, interval).completed
        return bars[-count:] if count > 0 else []
