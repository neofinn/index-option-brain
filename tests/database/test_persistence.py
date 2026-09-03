"""Persistence behaviour, against a real SQLite database.

Not mocks: the point of this layer is that rows survive a round trip, and a
mocked session proves nothing about a Numeric column or a dedup constraint.
SQLite is one of the two supported backends, so these are integration tests
of a real code path rather than a stand-in for Postgres.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from index_option_brain.brain.pipeline import QuantitativeBrain
from index_option_brain.contracts.feedback import Lesson, TradeFeedback
from index_option_brain.database.engine import Database, normalise_url
from index_option_brain.database.models import MarketSnapshot, OptionSnapshot
from index_option_brain.database.repository import (
    SnapshotRepository,
    SqlTradeMemoryRepository,
)

brain = QuantitativeBrain()


@pytest.fixture
async def database():
    db = Database.in_memory()
    await db.create_schema()
    yield db
    await db.aclose()


class TestUrlNormalisation:
    def test_a_synchronous_postgres_url_is_given_an_async_driver(self) -> None:
        """This is what people paste from a hosting dashboard, and without
        the rewrite it fails with an error about greenlets that says nothing
        about the cause."""
        assert normalise_url("postgresql://u:p@h/db").startswith(
            "postgresql+asyncpg://"
        )
        assert normalise_url("postgres://u:p@h/db").startswith("postgresql+asyncpg://")

    def test_sqlite_is_given_aiosqlite(self) -> None:
        assert normalise_url("sqlite:///x.db") == "sqlite+aiosqlite:///x.db"

    def test_an_already_async_url_is_left_alone(self) -> None:
        url = "postgresql+asyncpg://u:p@h/db"
        assert normalise_url(url) == url


class TestSnapshotRecording:
    async def test_a_state_and_its_chain_survive_the_round_trip(
        self, database: Database, uptrend_state
    ) -> None:
        async with database.session() as session:
            row = await SnapshotRepository(session).record_snapshot(uptrend_state)
            snapshot_id = row.id
            legs = row.option_legs

        assert legs == len(uptrend_state.options_state.chain)
        async with database.session() as session:
            stored = await session.get(MarketSnapshot, snapshot_id)
            assert stored is not None
            # A price is money: Numeric, not Float, or 23,914.45 stops being
            # equal to itself.
            assert Decimal(str(stored.spot)) == uptrend_state.index_state.quote.ltp
            assert stored.option_legs == legs

    async def test_recording_the_same_moment_twice_writes_one_row(
        self, database: Database, uptrend_state
    ) -> None:
        """The capture polls on a timer and restarts with the process, so it
        will be asked to record the same moment again. Dedup is on the feed's
        timestamp; on wall clock it would never match and every restart would
        double the corpus."""
        async with database.session() as session:
            repository = SnapshotRepository(session)
            first = await repository.record_snapshot(uptrend_state)
            second = await repository.record_snapshot(uptrend_state)
            assert first.id == second.id

        async with database.session() as session:
            coverage = await SnapshotRepository(session).chain_coverage(
                uptrend_state.index_symbol
            )
        assert coverage["snapshots"] == 1

    async def test_the_chain_can_be_skipped_for_a_dense_state_capture(
        self, database: Database, uptrend_state
    ) -> None:
        """~170 rows per chain, and a backtest wants many days at a
        comparable time rather than one day densely."""
        async with database.session() as session:
            await SnapshotRepository(session).record_snapshot(
                uptrend_state, with_chain=False
            )
        async with database.session() as session:
            legs = await session.execute(__import__("sqlalchemy").select(OptionSnapshot))
            assert list(legs.scalars()) == []

    async def test_greeks_are_stored_as_computed_not_recomputed(
        self, database: Database, uptrend_state
    ) -> None:
        """They were derived against a parity forward that needs the whole
        book. Only the mid survives in this table, so recomputing later would
        give numbers the system never acted on."""
        with_greeks = [
            q for q in uptrend_state.options_state.chain if q.greeks is not None
        ]
        if not with_greeks:
            pytest.skip("fixture chain carries no greeks")
        async with database.session() as session:
            await SnapshotRepository(session).record_snapshot(uptrend_state)

        import sqlalchemy

        async with database.session() as session:
            rows = await session.execute(
                sqlalchemy.select(OptionSnapshot).where(
                    OptionSnapshot.delta.is_not(None)
                )
            )
            stored = list(rows.scalars())
        assert len(stored) == len(with_greeks)

    async def test_an_unmeasured_reading_is_null_and_not_zero(
        self, database: Database, uptrend_state
    ) -> None:
        """A basis of zero says the futures are flat to carry; no basis says
        nobody looked. The column has to be able to say the second."""
        async with database.session() as session:
            row = await SnapshotRepository(session).record_snapshot(uptrend_state)
            snapshot_id = row.id

        async with database.session() as session:
            stored = await session.get(MarketSnapshot, snapshot_id)
            assert stored is not None
            if uptrend_state.options_state.forward is None:
                assert stored.forward is None
                assert stored.forward_excess_basis is None

    async def test_breadth_carries_its_own_timestamp(
        self, database: Database, uptrend_state
    ) -> None:
        """Breadth comes from the opening auction and freezes at 09:12. One
        timestamp on the row would make a 09:07 reading look as fresh as the
        14:00 spot beside it."""
        async with database.session() as session:
            row = await SnapshotRepository(session).record_snapshot(uptrend_state)
            if uptrend_state.constituent_state.quotes:
                assert row.breadth_observed_at is not None
            else:
                assert row.breadth_observed_at is None
                assert row.breadth_advances is None


class TestCycleRecording:
    async def test_the_reasoning_is_kept_not_just_the_verdict(
        self, database: Database, uptrend_state
    ) -> None:
        """Spec §31. A regime of TREND_DOWN at 0.29 is not reviewable; the
        same label with its evidence beside it is."""
        result = brain.run(uptrend_state)
        async with database.session() as session:
            row = await SnapshotRepository(session).record_cycle(result)
            assert row.regime == str(result.regime.regime)
            assert row.regime_evidence == list(result.regime.evidence)
            assert row.signal_evidence == list(result.signal.evidence)

    async def test_nothing_is_recorded_as_authorized_without_a_broker(
        self, database: Database, uptrend_state
    ) -> None:
        result = brain.run(uptrend_state)
        async with database.session() as session:
            row = await SnapshotRepository(session).record_cycle(result)
            assert row.is_authorized is False
            assert row.authorization_blocked_reason

    async def test_no_candidate_is_null_rather_than_an_empty_object(
        self, database: Database, uptrend_state
    ) -> None:
        """NULL means nothing was ranked; {} would mean a structure with no
        legs. They are different facts."""
        result = brain.run(uptrend_state)
        async with database.session() as session:
            row = await SnapshotRepository(session).record_cycle(result)
            if result.best_candidate is None:
                assert row.candidate is None

    async def test_recent_cycles_come_back_newest_first(
        self, database: Database, uptrend_state
    ) -> None:
        result = brain.run(uptrend_state)
        base = uptrend_state.timestamp
        async with database.session() as session:
            repository = SnapshotRepository(session)
            for offset in range(3):
                moved = result.model_copy(
                    update={
                        "state": uptrend_state.model_copy(
                            update={"timestamp": base + timedelta(minutes=offset)}
                        )
                    }
                )
                await repository.record_cycle(moved)

        async with database.session() as session:
            cycles = await SnapshotRepository(session).recent_cycles(
                uptrend_state.index_symbol
            )
        assert len(cycles) == 3
        assert cycles[0].observed_at > cycles[-1].observed_at


class TestLearningRecord:
    def _feedback(self, suffix: str) -> TradeFeedback:
        return TradeFeedback(
            feedback_id=f"fb-{suffix}",
            thesis_id="th-1",
            original_thesis="Banks leading a gap-up hold above max pain",
            scenario_id="sc-1",
            strategy="LONG_CALL",
            strike_summary="NIFTY 8-Sep 24100 CE x1",
            risk_summary="1 lot, max loss 5,879",
            expected_behavior="Extend past 24,050 with breadth intact",
            actual_behavior="Faded into the 24,200 wall",
            exit_reason="STOP",
            pnl=Decimal(-1900),
            thesis_confirmed=False,
            failure_reason="Basis premium decayed through the session",
            recorded_at=datetime.now(UTC),
        )

    async def test_feedback_round_trips(self, database: Database) -> None:
        async with database.session() as session:
            await SqlTradeMemoryRepository(session).save_feedback(self._feedback("a"))
        async with database.session() as session:
            history = await SqlTradeMemoryRepository(session).get_trade_history()
        assert len(history) == 1
        assert history[0].pnl == Decimal(-1900)
        assert history[0].thesis_confirmed is False

    async def test_a_lesson_arrives_unvalidated_and_unapproved(
        self, database: Database
    ) -> None:
        """Spec §20. `validated` is a fact about evidence, `approved_at` is a
        human act, and the pipeline that derives a lesson sets neither."""
        async with database.session() as session:
            await SqlTradeMemoryRepository(session).save_lesson(
                Lesson(
                    lesson_id="ls-1",
                    summary="Gap-up entries into a call wall underperform",
                    derived_from_feedback_ids=["fb-a", "fb-b"],
                    validated=True,  # even when the caller claims otherwise
                )
            )
        async with database.session() as session:
            lessons = await SqlTradeMemoryRepository(session).get_lessons()
        assert len(lessons) == 1
        assert lessons[0].validated is False
        assert lessons[0].approved_at is None
        assert lessons[0].supporting_trades == 2

    async def test_a_proposed_strategy_version_is_never_production(
        self, database: Database
    ) -> None:
        """The structural half of "no direct automatic production parameter
        mutation". There is no promote method on this repository."""
        async with database.session() as session:
            repository = SqlTradeMemoryRepository(session)
            row = await repository.propose_strategy_version(
                parameters={"min_long_leg_delta": 0.35}, lesson_ids=["ls-1"]
            )
            assert row.is_production is False
            assert row.approved_at is None
            assert await repository.production_version() is None
        assert not hasattr(SqlTradeMemoryRepository, "promote")

    async def test_position_history_raises_rather_than_returning_empty(
        self, database: Database
    ) -> None:
        """Positions have no table: the broker's response mapping is
        unverified, so the columns would be a guess. An empty list here would
        be read as "no positions"."""
        async with database.session() as session:
            with pytest.raises(NotImplementedError, match="guess"):
                await SqlTradeMemoryRepository(session).get_position_history("th-1")


class TestFailingSafe:
    async def test_reachability_is_checked_by_connecting_not_by_constructing(
        self,
    ) -> None:
        """SQLAlchemy connects lazily, so building a session against a dead
        database succeeds and the failure surfaces later inside code that
        assumed it had a working one."""
        broken = Database(url="postgresql+asyncpg://nobody@127.0.0.1:1/none")
        assert await broken.is_reachable() is False
        await broken.aclose()

    async def test_a_live_database_reports_reachable(
        self, database: Database
    ) -> None:
        assert await database.is_reachable() is True

    async def test_a_logged_url_carries_no_password(self) -> None:
        """An unreachable-database warning is exactly the line that gets
        pasted into an issue."""
        from index_option_brain.database.engine import _redact

        redacted = _redact("postgresql+asyncpg://user:s3cret@db.internal:5432/brain")
        assert "s3cret" not in redacted
        assert "db.internal:5432/brain" in redacted

    async def test_a_capture_write_failure_does_not_reach_the_caller(
        self, uptrend_state
    ) -> None:
        """Spec §29. A full disk must not stop the Execution Gate from
        gating, so the recorder owns an explicit try/except around its own
        writes rather than hiding one in the session helper."""
        from index_option_brain.capture import CaptureRecorder

        recorder = CaptureRecorder(
            database=Database(url="postgresql+asyncpg://nobody@127.0.0.1:1/none")
        )
        result = brain.run(uptrend_state)
        assert await recorder.record(result) is False
        assert recorder.stats.failures == 1
        assert recorder.stats.last_error
