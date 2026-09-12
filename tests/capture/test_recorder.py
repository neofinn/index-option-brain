"""Capture behaviour.

Two properties carry this component: it must never be able to fail an
analysis cycle, and it must not double-count on restart. Everything else it
does is bookkeeping.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from index_option_brain.brain.pipeline import QuantitativeBrain
from index_option_brain.capture import CaptureConfig, CaptureRecorder
from index_option_brain.contracts.enums import MarketSessionState
from index_option_brain.database.engine import Database

brain = QuantitativeBrain()


@pytest.fixture
async def recorder():
    database = Database.in_memory()
    yield CaptureRecorder(
        database=database,
        config=CaptureConfig(
            state_interval=timedelta(0), chain_interval=timedelta(0)
        ),
    )
    await database.aclose()


def at(state, **kw):
    return state.model_copy(update=kw)


class TestNeverFailsTheCaller:
    async def test_a_dead_database_returns_false_and_records_the_failure(
        self, uptrend_state
    ) -> None:
        recorder = CaptureRecorder(
            database=Database(url="postgresql+asyncpg://nobody@127.0.0.1:1/none")
        )
        assert await recorder.record(brain.run(uptrend_state)) is False
        assert recorder.stats.failures >= 1

    async def test_a_capture_failure_is_visible_rather_than_silent(
        self, uptrend_state
    ) -> None:
        """A capture that quietly stopped looks exactly like one with nothing
        to do, and one of those costs a year of corpus."""
        recorder = CaptureRecorder(
            database=Database(url="postgresql+asyncpg://nobody@127.0.0.1:1/none")
        )
        await recorder.record(brain.run(uptrend_state))
        assert recorder.stats.as_dict()["last_error"]


class TestIdempotency:
    async def test_the_same_feed_moment_is_recorded_once(
        self, recorder: CaptureRecorder, uptrend_state
    ) -> None:
        """The loop polls on a timer and restarts with the process."""
        result = brain.run(uptrend_state)
        assert await recorder.record(result) is True
        assert await recorder.record(result) is False
        assert recorder.stats.states_written == 1
        assert recorder.stats.duplicates_skipped == 1

    async def test_a_duplicate_does_not_advance_the_rate_limit(
        self, uptrend_state
    ) -> None:
        """If a repeated snapshot moved the clock forward, the next genuinely
        new moment would be skipped for being 'too soon'."""
        database = Database.in_memory()
        recorder = CaptureRecorder(
            database=database,
            config=CaptureConfig(
                state_interval=timedelta(minutes=1), chain_interval=timedelta(0)
            ),
        )
        base = brain.run(uptrend_state)
        await recorder.record(base)
        first_seen = recorder.stats.last_state_at

        await recorder.record(base)  # duplicate
        assert recorder.stats.last_state_at == first_seen

        moved = brain.run(
            at(uptrend_state, timestamp=uptrend_state.timestamp + timedelta(minutes=2))
        )
        assert await recorder.record(moved) is True
        await database.aclose()


class TestCadence:
    async def test_the_chain_is_written_less_often_than_the_state(
        self, uptrend_state
    ) -> None:
        """~170 rows per chain, and a backtest wants many days at a
        comparable time rather than one day densely."""
        database = Database.in_memory()
        recorder = CaptureRecorder(
            database=database,
            config=CaptureConfig(
                state_interval=timedelta(seconds=30),
                chain_interval=timedelta(minutes=10),
            ),
        )
        for minutes in (0, 1, 2, 3):
            await recorder.record(
                brain.run(
                    at(
                        uptrend_state,
                        timestamp=uptrend_state.timestamp + timedelta(minutes=minutes),
                    )
                )
            )
        assert recorder.stats.states_written == 4
        assert recorder.stats.chains_written == 1
        await database.aclose()

    async def test_a_closed_market_is_not_captured_by_default(
        self, recorder: CaptureRecorder, uptrend_state
    ) -> None:
        """A closed market republishes one snapshot indefinitely. The dedup
        key makes those writes harmless, but they make the corpus look denser
        than the information in it."""
        closed = at(uptrend_state, session_state=MarketSessionState.CLOSED)
        assert await recorder.record(brain.run(closed)) is False
        assert recorder.stats.states_written == 0


class TestCoverage:
    async def test_coverage_reports_sessions_not_just_rows(
        self, recorder: CaptureRecorder, uptrend_state
    ) -> None:
        """"Building a corpus" and "three days of one expiry" look identical
        without it, and only one supports a backtest."""
        await recorder.record(brain.run(uptrend_state))
        coverage = await recorder.coverage(uptrend_state.index_symbol)
        assert coverage["snapshots"] == 1
        assert coverage["sessions"] == 1
        assert coverage["legs"] > 0
        assert coverage["first"] == coverage["last"]

    async def test_coverage_of_an_empty_store_is_zero_not_absent(
        self, recorder: CaptureRecorder
    ) -> None:
        coverage = await recorder.coverage("NIFTY")
        assert coverage["snapshots"] == 0
        assert coverage["first"] is None
