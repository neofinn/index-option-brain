"""Persisting observed bars across restarts.

Bars are expensive here — NSE serves no history, so a week of 5-minute bars
is a week of uptime. The tests that matter are the ones about what the store
*refuses* to load: a series seeded from a truncated file is a wrong series,
and no indicator downstream can tell.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from index_option_brain.contracts.enums import BarInterval
from index_option_brain.contracts.instruments import Bar
from index_option_brain.data.bar_store import SCHEMA_VERSION, BarStore

OPEN_UTC = datetime(2026, 9, 2, 3, 45, tzinfo=UTC)


def bars(count: int = 3) -> list[Bar]:
    return [
        Bar(
            timestamp=OPEN_UTC + timedelta(minutes=5 * n),
            open=Decimal("23900.50"),
            high=Decimal("23950.75"),
            low=Decimal("23880.25"),
            close=Decimal("23910.05"),
            volume=0,
        )
        for n in range(count)
    ]


@pytest.fixture
def store(tmp_path) -> BarStore:
    return BarStore(tmp_path / "bars")


class TestRoundTrip:
    def test_bars_survive_a_save_and_load(self, store):
        store.save("NIFTY", BarInterval.MINUTE_5, bars())
        restored = store.load("NIFTY", BarInterval.MINUTE_5)
        assert len(restored) == 3

    def test_decimal_precision_is_preserved(self, store):
        """Prices go through as strings. A float round trip would quietly
        move the last paisa on every bar."""
        store.save("NIFTY", BarInterval.MINUTE_5, bars(1))
        restored = store.load("NIFTY", BarInterval.MINUTE_5)
        assert restored[0].open == Decimal("23900.50")
        assert restored[0].high == Decimal("23950.75")

    def test_timestamps_stay_timezone_aware(self, store):
        store.save("NIFTY", BarInterval.MINUTE_5, bars(1))
        restored = store.load("NIFTY", BarInterval.MINUTE_5)
        assert restored[0].timestamp == OPEN_UTC
        assert restored[0].timestamp.tzinfo is not None

    def test_bars_come_back_oldest_first(self, store):
        store.save("NIFTY", BarInterval.MINUTE_5, list(reversed(bars(5))))
        restored = store.load("NIFTY", BarInterval.MINUTE_5)
        assert [b.timestamp for b in restored] == sorted(b.timestamp for b in restored)

    def test_symbols_and_intervals_are_separate_files(self, store):
        store.save("NIFTY", BarInterval.MINUTE_5, bars(2))
        store.save("NIFTY", BarInterval.DAY, bars(4))
        store.save("BANKNIFTY", BarInterval.MINUTE_5, bars(6))
        assert len(store.load("NIFTY", BarInterval.MINUTE_5)) == 2
        assert len(store.load("NIFTY", BarInterval.DAY)) == 4
        assert len(store.load("BANKNIFTY", BarInterval.MINUTE_5)) == 6

    def test_the_save_time_is_recorded(self, store):
        store.save("NIFTY", BarInterval.MINUTE_5, bars(1))
        saved = store.saved_at("NIFTY", BarInterval.MINUTE_5)
        assert saved is not None
        assert saved.tzinfo is not None


class TestWhatItRefusesToLoad:
    """Starting cold costs a session of bars. A mis-seeded series is a wrong
    answer no downstream indicator can detect, so every failure path returns
    nothing rather than something."""

    def test_a_missing_file_loads_nothing(self, store):
        assert store.load("NIFTY", BarInterval.MINUTE_5) == []

    def test_corrupt_json_loads_nothing(self, store):
        store.save("NIFTY", BarInterval.MINUTE_5, bars())
        store.path_for("NIFTY", BarInterval.MINUTE_5).write_text("{not json")
        assert store.load("NIFTY", BarInterval.MINUTE_5) == []

    def test_a_truncated_file_loads_nothing(self, store):
        """Half a bar series is not a shorter series, it is a wrong one."""
        store.save("NIFTY", BarInterval.MINUTE_5, bars(10))
        path = store.path_for("NIFTY", BarInterval.MINUTE_5)
        path.write_text(path.read_text()[: len(path.read_text()) // 2])
        assert store.load("NIFTY", BarInterval.MINUTE_5) == []

    def test_one_malformed_bar_condemns_the_file(self, store):
        """A gap in the middle of a series is worse than no series, because
        it is invisible."""
        store.save("NIFTY", BarInterval.MINUTE_5, bars(4))
        path = store.path_for("NIFTY", BarInterval.MINUTE_5)
        payload = json.loads(path.read_text())
        del payload["bars"][2]["high"]
        path.write_text(json.dumps(payload))
        assert store.load("NIFTY", BarInterval.MINUTE_5) == []

    def test_an_older_schema_is_discarded_not_migrated(self, store):
        store.save("NIFTY", BarInterval.MINUTE_5, bars())
        path = store.path_for("NIFTY", BarInterval.MINUTE_5)
        payload = json.loads(path.read_text())
        payload["schema_version"] = SCHEMA_VERSION - 1
        path.write_text(json.dumps(payload))
        assert store.load("NIFTY", BarInterval.MINUTE_5) == []

    def test_a_file_holding_the_wrong_interval_is_rejected(self, store):
        """Otherwise a mislabelled file would seed 5-minute bars into a daily
        series, and every level computed from it would be wrong by a factor of
        the interval."""
        store.save("NIFTY", BarInterval.MINUTE_5, bars())
        path = store.path_for("NIFTY", BarInterval.MINUTE_5)
        payload = json.loads(path.read_text())
        payload["interval"] = str(BarInterval.DAY)
        path.write_text(json.dumps(payload))
        assert store.load("NIFTY", BarInterval.MINUTE_5) == []

    def test_naive_timestamps_are_rejected(self, store):
        """A naive timestamp would make every session boundary computed from
        it depend on the host's timezone."""
        store.save("NIFTY", BarInterval.MINUTE_5, bars(2))
        path = store.path_for("NIFTY", BarInterval.MINUTE_5)
        payload = json.loads(path.read_text())
        payload["bars"][0]["timestamp"] = "2026-09-02T03:45:00"
        path.write_text(json.dumps(payload))
        assert store.load("NIFTY", BarInterval.MINUTE_5) == []


class TestAtomicWrites:
    def test_no_temporary_files_are_left_behind(self, store):
        """The process is killed at arbitrary moments; a stray half-written
        file is exactly the one that would be read next."""
        store.save("NIFTY", BarInterval.MINUTE_5, bars())
        leftovers = [p for p in store.directory.iterdir() if p.suffix == ".tmp"]
        assert leftovers == []

    def test_an_overwrite_replaces_cleanly(self, store):
        store.save("NIFTY", BarInterval.MINUTE_5, bars(2))
        store.save("NIFTY", BarInterval.MINUTE_5, bars(7))
        assert len(store.load("NIFTY", BarInterval.MINUTE_5)) == 7

    def test_the_directory_is_created_on_demand(self, tmp_path):
        store = BarStore(tmp_path / "deep" / "nested")
        store.save("NIFTY", BarInterval.DAY, bars(1))
        assert store.load("NIFTY", BarInterval.DAY)


class TestAggregatorIntegration:
    async def test_a_restart_keeps_the_observed_bars(self, store):
        """The whole point. Without this, every deploy is a decision to go
        blind for a session."""
        from index_option_brain.data.bar_aggregator import AggregatingIndexAdapter
        from tests.data.test_bar_aggregator import _StubIndexSource, quote

        quotes = [
            quote(OPEN_UTC + timedelta(minutes=m), f"{23900 + m}") for m in range(12)
        ]
        first = AggregatingIndexAdapter(
            _StubIndexSource(quotes), intervals=(BarInterval.MINUTE_5,), store=store
        )
        for _ in range(12):
            await first.get_index_quote("NIFTY")
        observed = await first.get_index_bars("NIFTY", BarInterval.MINUTE_5, 99)
        assert len(observed) == 2
        assert first.persist("NIFTY") == 1

        # A fresh process, same store.
        second = AggregatingIndexAdapter(
            _StubIndexSource([]), intervals=(BarInterval.MINUTE_5,), store=store
        )
        restored = await second.get_index_bars("NIFTY", BarInterval.MINUTE_5, 99)
        assert len(restored) == 2
        assert restored[0].open == observed[0].open

    async def test_without_a_store_nothing_is_persisted(self):
        from index_option_brain.data.bar_aggregator import AggregatingIndexAdapter
        from tests.data.test_bar_aggregator import _StubIndexSource

        adapter = AggregatingIndexAdapter(_StubIndexSource([]))
        assert adapter.persist() == 0

    async def test_an_unreadable_snapshot_starts_cold_rather_than_wrong(self, store):
        from index_option_brain.data.bar_aggregator import AggregatingIndexAdapter
        from tests.data.test_bar_aggregator import _StubIndexSource

        store.save("NIFTY", BarInterval.MINUTE_5, bars(5))
        store.path_for("NIFTY", BarInterval.MINUTE_5).write_text("garbage")
        adapter = AggregatingIndexAdapter(
            _StubIndexSource([]), intervals=(BarInterval.MINUTE_5,), store=store
        )
        assert await adapter.get_index_bars("NIFTY", BarInterval.MINUTE_5, 99) == []
