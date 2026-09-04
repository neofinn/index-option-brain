"""Tests for building bars from live snapshots.

The interesting cases are all about what the aggregator *refuses* to build. A
naive version of this passes a happy-path test and then quietly manufactures a
week of flat candles over a long weekend, or labels a half-observed window as a
complete bar. Those are the tests that matter.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from index_option_brain.contracts.enums import BarInterval
from index_option_brain.contracts.instruments import Bar, IndexQuote
from index_option_brain.data.adapters.base import DataAdapterError, IndexDataAdapter
from index_option_brain.data.bar_aggregator import (
    IST,
    AggregatingIndexAdapter,
    LiveBarAggregator,
    bucket_start,
    in_session,
    interval_seconds,
    session_close,
    session_open,
)

# Wednesday 02-Sep-2026. 09:15 IST is 03:45 UTC and 15:30 IST is 10:00 UTC.
OPEN_UTC = datetime(2026, 9, 2, 3, 45, tzinfo=UTC)
CLOSE_UTC = datetime(2026, 9, 2, 10, 0, tzinfo=UTC)


def quote(
    moment: datetime,
    ltp: str,
    *,
    open_: str = "23858",
    high: str = "23930",
    low: str = "23780",
    previous_close: str = "24055.8",
) -> IndexQuote:
    return IndexQuote(
        symbol="NIFTY",
        timestamp=moment,
        ltp=Decimal(ltp),
        open=Decimal(open_),
        high=Decimal(high),
        low=Decimal(low),
        previous_close=Decimal(previous_close),
    )


class TestSessionHelpers:
    def test_session_open_and_close_in_utc(self):
        moment = datetime(2026, 9, 2, 6, 0, tzinfo=UTC)
        assert session_open(moment) == OPEN_UTC
        assert session_close(moment) == CLOSE_UTC

    @pytest.mark.parametrize(
        ("moment", "expected"),
        [
            (OPEN_UTC, True),
            (OPEN_UTC - timedelta(seconds=1), False),
            (CLOSE_UTC, True),
            (CLOSE_UTC + timedelta(seconds=1), False),
            (datetime(2026, 9, 2, 6, 0, tzinfo=UTC), True),
        ],
    )
    def test_session_bounds_are_inclusive_of_open_and_close(self, moment, expected):
        assert in_session(moment) is expected

    def test_weekends_are_out_of_session(self):
        saturday = datetime(2026, 9, 5, 6, 0, tzinfo=UTC)
        sunday = datetime(2026, 9, 6, 6, 0, tzinfo=UTC)
        assert not in_session(saturday)
        assert not in_session(sunday)

    def test_buckets_are_anchored_to_the_session_open(self):
        """09:17 IST belongs to the 09:15 five-minute bar."""
        moment = OPEN_UTC + timedelta(minutes=2)
        assert bucket_start(moment, BarInterval.MINUTE_5) == OPEN_UTC

    def test_the_first_bucket_starts_exactly_at_the_open(self):
        assert bucket_start(OPEN_UTC, BarInterval.MINUTE_5) == OPEN_UTC
        assert bucket_start(OPEN_UTC, BarInterval.MINUTE_15) == OPEN_UTC

    def test_bucket_boundaries_step_by_the_interval(self):
        moment = OPEN_UTC + timedelta(minutes=7)
        assert bucket_start(moment, BarInterval.MINUTE_5) == OPEN_UTC + timedelta(
            minutes=5
        )
        assert bucket_start(moment, BarInterval.MINUTE_15) == OPEN_UTC

    def test_the_last_bucket_ends_at_the_close(self):
        """15:25-15:30 is a whole bar, so the session divides evenly and no
        stub candle is left at the end of the day."""
        last = bucket_start(CLOSE_UTC - timedelta(seconds=1), BarInterval.MINUTE_5)
        assert last + timedelta(minutes=5) == CLOSE_UTC

    def test_daily_has_no_intraday_interval(self):
        with pytest.raises(DataAdapterError, match="not an intraday interval"):
            interval_seconds(BarInterval.DAY)


class TestIntradayAggregation:
    def test_a_bucket_completes_when_the_next_one_starts(self):
        agg = LiveBarAggregator(BarInterval.MINUTE_5)
        assert agg.observe(quote(OPEN_UTC, "23900")) is None
        assert agg.observe(quote(OPEN_UTC + timedelta(minutes=2), "23920")) is None
        bar = agg.observe(quote(OPEN_UTC + timedelta(minutes=5), "23910"))
        assert bar is not None
        assert bar.timestamp == OPEN_UTC
        assert bar.open == Decimal(23900)
        assert bar.close == Decimal(23920)

    def test_high_and_low_are_the_observed_extremes(self):
        agg = LiveBarAggregator(BarInterval.MINUTE_5)
        for offset, price in ((0, "23900"), (1, "23945"), (2, "23880"), (3, "23910")):
            agg.observe(quote(OPEN_UTC + timedelta(minutes=offset), price))
        agg.observe(quote(OPEN_UTC + timedelta(minutes=5), "23915"))
        bar = agg.completed[0]
        assert bar.high == Decimal(23945)
        assert bar.low == Decimal(23880)
        assert bar.open == Decimal(23900)
        assert bar.close == Decimal(23910)

    def test_the_forming_bar_is_never_in_completed(self):
        """The adapter contract forbids a partial candle here: the brains read
        the last bar as a finished period."""
        agg = LiveBarAggregator(BarInterval.MINUTE_5)
        agg.observe(quote(OPEN_UTC, "23900"))
        assert agg.completed == []
        assert agg.forming is not None
        assert agg.forming.timestamp == OPEN_UTC

    def test_the_forming_bar_is_available_for_display(self):
        agg = LiveBarAggregator(BarInterval.MINUTE_5)
        agg.observe(quote(OPEN_UTC, "23900"))
        agg.observe(quote(OPEN_UTC + timedelta(minutes=1), "23930"))
        forming = agg.forming
        assert forming is not None
        assert forming.high == Decimal(23930)

    def test_volume_is_zero_because_nse_publishes_none(self):
        """Zero means "not measured". No indicator reads index volume, and a
        fabricated figure would be worse than an absent one."""
        agg = LiveBarAggregator(BarInterval.MINUTE_5)
        agg.observe(quote(OPEN_UTC, "23900"))
        agg.observe(quote(OPEN_UTC + timedelta(minutes=5), "23910"))
        assert agg.completed[0].volume == 0

    def test_several_buckets_accumulate_in_order(self):
        agg = LiveBarAggregator(BarInterval.MINUTE_5)
        for minute in range(21):
            agg.observe(quote(OPEN_UTC + timedelta(minutes=minute), f"{23900 + minute}"))
        bars = agg.completed
        assert len(bars) == 4
        assert [b.timestamp for b in bars] == [
            OPEN_UTC + timedelta(minutes=5 * i) for i in range(4)
        ]
        assert bars == sorted(bars, key=lambda b: b.timestamp)

    def test_max_bars_evicts_the_oldest(self):
        agg = LiveBarAggregator(BarInterval.MINUTE_1, max_bars=3)
        for minute in range(10):
            agg.observe(quote(OPEN_UTC + timedelta(minutes=minute), "23900"))
        bars = agg.completed
        assert len(bars) == 3
        assert bars[-1].timestamp == OPEN_UTC + timedelta(minutes=8)


class TestWhatItRefusesToBuild:
    def test_a_bucket_whose_open_was_missed_is_discarded(self):
        """A five-minute bar first observed two minutes in has no opening
        price. Labelling a later print as the open misstates the candle in
        exactly the direction the market moved."""
        agg = LiveBarAggregator(BarInterval.MINUTE_5)
        agg.observe(quote(OPEN_UTC + timedelta(minutes=2), "23920"))
        agg.observe(quote(OPEN_UTC + timedelta(minutes=5), "23930"))
        assert agg.completed == []
        assert agg.stats.bars_discarded_partial == 1

    def test_a_bucket_observed_within_the_lag_allowance_is_kept(self):
        """A feed that prints every few seconds will not land exactly on the
        boundary, so a small lag is tolerated — a quarter of the interval."""
        agg = LiveBarAggregator(BarInterval.MINUTE_5)
        agg.observe(quote(OPEN_UTC + timedelta(seconds=30), "23900"))
        agg.observe(quote(OPEN_UTC + timedelta(minutes=5), "23930"))
        assert len(agg.completed) == 1
        assert agg.stats.bars_discarded_partial == 0

    def test_out_of_session_snapshots_are_ignored(self):
        """The endpoint answers all night with the closing price. Folding that
        in would produce hundreds of flat bars and a fictitious ATR."""
        agg = LiveBarAggregator(BarInterval.MINUTE_5)
        for hour in range(11, 20):
            agg.observe(quote(datetime(2026, 9, 2, hour, 0, tzinfo=UTC), "23914.45"))
        assert agg.completed == []
        assert agg.forming is None
        assert agg.stats.out_of_session == 9

    def test_a_weekend_produces_no_bars(self):
        agg = LiveBarAggregator(BarInterval.MINUTE_5)
        saturday = datetime(2026, 9, 5, 6, 0, tzinfo=UTC)
        for minute in range(0, 60, 5):
            agg.observe(quote(saturday + timedelta(minutes=minute), "23914.45"))
        assert agg.completed == []

    def test_a_repeated_snapshot_is_discarded_as_stale(self):
        """NSE's timestamp advances on its own schedule, so a poll loop reads
        the same snapshot repeatedly. Deduping on the payload's timestamp is
        what makes that harmless."""
        agg = LiveBarAggregator(BarInterval.MINUTE_5)
        agg.observe(quote(OPEN_UTC, "23900"))
        for _ in range(5):
            agg.observe(quote(OPEN_UTC, "23900"))
        assert agg.stats.stale_snapshots == 5
        assert agg.stats.observations == 6

    def test_an_out_of_order_snapshot_is_discarded(self):
        agg = LiveBarAggregator(BarInterval.MINUTE_5)
        agg.observe(quote(OPEN_UTC + timedelta(minutes=2), "23900"))
        assert agg.observe(quote(OPEN_UTC + timedelta(minutes=1), "23950")) is None
        assert agg.stats.stale_snapshots == 1
        forming = agg.forming
        assert forming is not None
        assert forming.high == Decimal(23900)

    def test_a_holiday_needs_no_calendar(self):
        """On a holiday the exchange keeps serving the previous session's data
        with its original timestamp, so it is rejected as non-advancing. No
        holiday list is needed, and none can therefore go out of date."""
        agg = LiveBarAggregator(BarInterval.MINUTE_5)
        friday_close = datetime(2026, 9, 4, 10, 0, tzinfo=UTC)
        agg.observe(quote(friday_close, "23914.45"))
        before = agg.stats.stale_snapshots
        for _ in range(20):
            agg.observe(quote(friday_close, "23914.45"))
        assert agg.stats.stale_snapshots == before + 20
        assert agg.completed == []

    def test_a_feed_outage_is_counted_not_filled(self):
        """Nothing is interpolated across a gap. The hole is recorded so a
        caller can tell a short history from a gappy one — indicators read a
        bar list as contiguous, and a hole silently changes what ATR means."""
        agg = LiveBarAggregator(BarInterval.MINUTE_5)
        agg.observe(quote(OPEN_UTC, "23900"))
        agg.observe(quote(OPEN_UTC + timedelta(minutes=20), "23950"))
        assert len(agg.completed) == 1
        assert agg.stats.missing_buckets == 3
        assert agg.stats.has_gaps

    def test_an_overnight_break_is_not_an_outage(self):
        """The gap between one session's close and the next open is not a feed
        failure, and counting it would make every multi-day series look
        broken."""
        agg = LiveBarAggregator(BarInterval.MINUTE_5)
        agg.observe(quote(CLOSE_UTC - timedelta(minutes=5), "23900"))
        next_open = datetime(2026, 9, 3, 3, 45, tzinfo=UTC)
        agg.observe(quote(next_open, "23800"))
        assert agg.stats.missing_buckets == 0
        assert not agg.stats.has_gaps


class TestSessionClose:
    def test_closing_the_session_completes_the_last_bar(self):
        agg = LiveBarAggregator(BarInterval.MINUTE_5)
        agg.observe(quote(CLOSE_UTC - timedelta(minutes=5), "23900"))
        agg.observe(quote(CLOSE_UTC - timedelta(minutes=1), "23910"))
        bar = agg.close_session()
        assert bar is not None
        assert bar.close == Decimal(23910)
        assert len(agg.completed) == 1

    def test_closing_twice_is_harmless(self):
        agg = LiveBarAggregator(BarInterval.MINUTE_5)
        agg.observe(quote(OPEN_UTC, "23900"))
        agg.close_session()
        assert agg.close_session() is None
        assert len(agg.completed) == 1

    def test_closing_discards_a_partial_opening(self):
        agg = LiveBarAggregator(BarInterval.MINUTE_5)
        agg.observe(quote(OPEN_UTC + timedelta(minutes=3), "23900"))
        assert agg.close_session() is None
        assert agg.completed == []

    def test_nothing_reads_the_wall_clock(self):
        """`close_session` is called explicitly rather than inferred from a
        clock, so the same code path serves live, replay and backtest."""
        agg = LiveBarAggregator(BarInterval.MINUTE_5)
        agg.observe(quote(datetime(2019, 4, 3, 4, 0, tzinfo=UTC), "11500"))
        bar = agg.close_session()
        assert bar is not None
        assert bar.timestamp.year == 2019


class TestDailyAggregation:
    def test_the_daily_bar_uses_the_exchange_open_high_and_low(self):
        """Observed extremes would only cover the window this process was
        connected for. The exchange's figures cover the whole session,
        including before it started watching."""
        agg = LiveBarAggregator(BarInterval.DAY)
        agg.observe(
            quote(
                datetime(2026, 9, 2, 9, 0, tzinfo=UTC),
                "23914.45",
                open_="23858",
                high="23930",
                low="23780",
            )
        )
        forming = agg.forming
        assert forming is not None
        assert forming.open == Decimal(23858)
        assert forming.high == Decimal(23930)
        assert forming.low == Decimal(23780)
        assert forming.close == Decimal("23914.45")

    def test_the_daily_bar_is_stamped_at_the_session_open(self):
        agg = LiveBarAggregator(BarInterval.DAY)
        agg.observe(quote(datetime(2026, 9, 2, 9, 0, tzinfo=UTC), "23914.45"))
        forming = agg.forming
        assert forming is not None
        assert forming.timestamp == OPEN_UTC
        assert forming.timestamp.astimezone(IST).time().hour == 9

    def test_a_new_session_completes_the_previous_day(self):
        agg = LiveBarAggregator(BarInterval.DAY)
        agg.observe(quote(datetime(2026, 9, 2, 9, 0, tzinfo=UTC), "23914.45"))
        bar = agg.observe(
            quote(datetime(2026, 9, 3, 4, 0, tzinfo=UTC), "23800", open_="23890")
        )
        assert bar is not None
        assert bar.timestamp == OPEN_UTC
        assert bar.close == Decimal("23914.45")

    def test_a_glitched_snapshot_cannot_shrink_the_bar(self):
        """The exchange's high and low are cumulative, so a snapshot reporting
        a lower high is bad data rather than a new reading."""
        agg = LiveBarAggregator(BarInterval.DAY)
        agg.observe(
            quote(datetime(2026, 9, 2, 5, 0, tzinfo=UTC), "23900", high="23950", low="23800")
        )
        agg.observe(
            quote(datetime(2026, 9, 2, 6, 0, tzinfo=UTC), "23910", high="23800", low="23900")
        )
        forming = agg.forming
        assert forming is not None
        assert forming.high == Decimal(23950)
        assert forming.low == Decimal(23800)

    def test_a_pre_open_snapshot_is_ignored(self):
        """Before 09:15 the endpoint reports the previous session, and folding
        it into today would duplicate yesterday."""
        agg = LiveBarAggregator(BarInterval.DAY)
        agg.observe(quote(datetime(2026, 9, 2, 3, 0, tzinfo=UTC), "24055.8"))
        assert agg.forming is None
        assert agg.stats.out_of_session == 1

    def test_a_post_close_snapshot_still_updates_the_day(self):
        """The 15:40 chain timestamp is after the close but is still today's
        data — the day's bar is not final until the next session opens."""
        agg = LiveBarAggregator(BarInterval.DAY)
        agg.observe(quote(datetime(2026, 9, 2, 6, 0, tzinfo=UTC), "23900"))
        agg.observe(quote(datetime(2026, 9, 2, 10, 10, tzinfo=UTC), "23914.45"))
        forming = agg.forming
        assert forming is not None
        assert forming.close == Decimal("23914.45")

    def test_a_weekend_snapshot_is_ignored(self):
        agg = LiveBarAggregator(BarInterval.DAY)
        agg.observe(quote(datetime(2026, 9, 5, 6, 0, tzinfo=UTC), "23914.45"))
        assert agg.forming is None


class TestSeeding:
    def test_seeded_history_is_kept_oldest_first(self):
        agg = LiveBarAggregator(BarInterval.DAY)
        bars = [
            Bar(
                timestamp=OPEN_UTC - timedelta(days=n),
                open=Decimal(23000),
                high=Decimal(23100),
                low=Decimal(22900),
                close=Decimal(23050),
            )
            for n in (3, 1, 2)
        ]
        agg.seed(bars)
        stamps = [b.timestamp for b in agg.completed]
        assert stamps == sorted(stamps)
        assert agg.stats.seeded_bars == 3

    def test_seeding_respects_max_bars(self):
        agg = LiveBarAggregator(BarInterval.MINUTE_5, max_bars=2)
        bars = [
            Bar(
                timestamp=OPEN_UTC + timedelta(minutes=5 * n),
                open=Decimal(23000),
                high=Decimal(23100),
                low=Decimal(22900),
                close=Decimal(23050),
            )
            for n in range(5)
        ]
        agg.seed(bars)
        assert len(agg.completed) == 2
        assert agg.completed[-1].timestamp == OPEN_UTC + timedelta(minutes=20)

    def test_live_observation_extends_seeded_history(self):
        """The warm-start path: a broker supplies history, this extends it, so
        a fresh process is not blind for the first hour."""
        agg = LiveBarAggregator(BarInterval.MINUTE_5)
        agg.seed(
            [
                Bar(
                    timestamp=OPEN_UTC,
                    open=Decimal(23800),
                    high=Decimal(23850),
                    low=Decimal(23790),
                    close=Decimal(23840),
                )
            ]
        )
        agg.observe(quote(OPEN_UTC + timedelta(minutes=5), "23900"))
        agg.observe(quote(OPEN_UTC + timedelta(minutes=10), "23920"))
        bars = agg.completed
        assert len(bars) == 2
        assert bars[0].close == Decimal(23840)
        assert bars[1].open == Decimal(23900)

    def test_a_snapshot_older_than_seeded_history_is_rejected(self):
        agg = LiveBarAggregator(BarInterval.MINUTE_5)
        agg.seed(
            [
                Bar(
                    timestamp=OPEN_UTC + timedelta(minutes=30),
                    open=Decimal(23800),
                    high=Decimal(23850),
                    low=Decimal(23790),
                    close=Decimal(23840),
                )
            ]
        )
        assert agg.observe(quote(OPEN_UTC, "23900")) is None
        assert agg.stats.stale_snapshots == 1


class _StubIndexSource(IndexDataAdapter):
    """A minimal source, so the wrapper is tested and not the feed."""

    def __init__(self, quotes: list[IndexQuote]) -> None:
        self._quotes = list(quotes)
        self.spec_calls = 0

    async def get_index_spec(self, symbol: str):
        from index_option_brain.contracts.instruments import IndexSpec

        self.spec_calls += 1
        return IndexSpec(
            symbol=symbol, name=symbol, lot_size=75, tick_size=Decimal("0.05")
        )

    async def get_index_quote(self, symbol: str) -> IndexQuote:
        return self._quotes.pop(0)

    async def get_index_bars(self, symbol: str, interval: BarInterval, count: int):
        raise DataAdapterError("this source has no history")


class TestAggregatingAdapter:
    async def test_reading_quotes_builds_bars(self):
        """Every snapshot read for analysis doubles as a bar observation, so
        no separate polling loop is needed."""
        quotes = [
            quote(OPEN_UTC + timedelta(minutes=minute), f"{23900 + minute}")
            for minute in range(12)
        ]
        adapter = AggregatingIndexAdapter(
            _StubIndexSource(quotes), intervals=(BarInterval.MINUTE_5,)
        )
        for _ in range(12):
            await adapter.get_index_quote("NIFTY")
        bars = await adapter.get_index_bars("NIFTY", BarInterval.MINUTE_5, 10)
        assert len(bars) == 2
        assert bars[0].open == Decimal(23900)

    async def test_a_cold_start_returns_no_bars_rather_than_raising(self):
        """The indicators return None on insufficient data, so a short series
        degrades the analysis honestly. Padded bars would produce a confident
        wrong answer instead."""
        adapter = AggregatingIndexAdapter(_StubIndexSource([]))
        assert await adapter.get_index_bars("NIFTY", BarInterval.DAY, 20) == []

    async def test_it_serves_fewer_bars_than_requested(self):
        quotes = [
            quote(OPEN_UTC + timedelta(minutes=minute), "23900")
            for minute in range(12)
        ]
        adapter = AggregatingIndexAdapter(
            _StubIndexSource(quotes), intervals=(BarInterval.MINUTE_5,)
        )
        for _ in range(12):
            await adapter.get_index_quote("NIFTY")
        assert len(await adapter.get_index_bars("NIFTY", BarInterval.MINUTE_5, 500)) == 2

    async def test_it_returns_the_newest_bars_when_more_exist(self):
        quotes = [
            quote(OPEN_UTC + timedelta(minutes=minute), f"{23900 + minute}")
            for minute in range(21)
        ]
        adapter = AggregatingIndexAdapter(
            _StubIndexSource(quotes), intervals=(BarInterval.MINUTE_5,)
        )
        for _ in range(21):
            await adapter.get_index_quote("NIFTY")
        bars = await adapter.get_index_bars("NIFTY", BarInterval.MINUTE_5, 2)
        assert len(bars) == 2
        assert bars[-1].timestamp == OPEN_UTC + timedelta(minutes=15)

    async def test_the_spec_is_delegated_to_the_source(self):
        source = _StubIndexSource([])
        adapter = AggregatingIndexAdapter(source)
        spec = await adapter.get_index_spec("NIFTY")
        assert spec.lot_size == 75
        assert source.spec_calls == 1

    async def test_an_untracked_interval_raises_with_what_is_tracked(self):
        adapter = AggregatingIndexAdapter(
            _StubIndexSource([]), intervals=(BarInterval.MINUTE_5,)
        )
        with pytest.raises(DataAdapterError) as exc:
            await adapter.get_index_bars("NIFTY", BarInterval.MINUTE_15, 10)
        assert "not being aggregated" in str(exc.value)
        assert "5m" in str(exc.value)

    async def test_intervals_are_aggregated_independently(self):
        quotes = [
            quote(OPEN_UTC + timedelta(minutes=minute), f"{23900 + minute}")
            for minute in range(31)
        ]
        adapter = AggregatingIndexAdapter(
            _StubIndexSource(quotes),
            intervals=(BarInterval.MINUTE_5, BarInterval.MINUTE_15),
        )
        for _ in range(31):
            await adapter.get_index_quote("NIFTY")
        assert len(await adapter.get_index_bars("NIFTY", BarInterval.MINUTE_5, 99)) == 6
        assert len(await adapter.get_index_bars("NIFTY", BarInterval.MINUTE_15, 99)) == 2

    async def test_symbols_are_aggregated_independently(self):
        adapter = AggregatingIndexAdapter(
            _StubIndexSource([]), intervals=(BarInterval.MINUTE_5,)
        )
        nifty = adapter.aggregator("NIFTY", BarInterval.MINUTE_5)
        banknifty = adapter.aggregator("BANKNIFTY", BarInterval.MINUTE_5)
        assert nifty is not banknifty
        assert adapter.aggregator("nifty", BarInterval.MINUTE_5) is nifty

    async def test_stats_report_coverage(self):
        """A caller must be able to tell a short history from a gappy one."""
        quotes = [quote(OPEN_UTC, "23900"), quote(OPEN_UTC + timedelta(minutes=20), "23950")]
        adapter = AggregatingIndexAdapter(
            _StubIndexSource(quotes), intervals=(BarInterval.MINUTE_5,)
        )
        for _ in range(2):
            await adapter.get_index_quote("NIFTY")
        stats = adapter.stats("NIFTY", BarInterval.MINUTE_5)
        assert stats.observations == 2
        assert stats.missing_buckets == 3
        assert stats.has_gaps
        assert stats.first_observation == OPEN_UTC

    async def test_seeded_bars_are_served_through_the_adapter(self):
        adapter = AggregatingIndexAdapter(
            _StubIndexSource([]), intervals=(BarInterval.DAY,)
        )
        adapter.seed(
            "NIFTY",
            BarInterval.DAY,
            [
                Bar(
                    timestamp=OPEN_UTC - timedelta(days=1),
                    open=Decimal(23800),
                    high=Decimal(24000),
                    low=Decimal(23700),
                    close=Decimal(23950),
                )
            ],
        )
        bars = await adapter.get_index_bars("NIFTY", BarInterval.DAY, 10)
        assert len(bars) == 1
        assert bars[0].close == Decimal(23950)
