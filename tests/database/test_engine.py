"""Engine and session plumbing."""

from __future__ import annotations

from index_option_brain.database.engine import Database


class TestConcurrentWriters:
    """The deployment now has two writing processes: the engine capturing
    snapshots and the TradingView receiver recording accepted alerts."""

    def test_sqlite_waits_for_a_lock_instead_of_failing_immediately(self) -> None:
        """SQLite's default is to fail with "database is locked" rather
        than to wait. Without a busy timeout, an alert arriving during a
        chain capture is dropped for a lock that would have cleared in
        milliseconds — and the alert path answers 503 on a failed write."""
        database = Database.sqlite("var/test-busy-timeout.sqlite")
        assert database._connect_args(database.url) == {"timeout": 30.0}

    def test_postgres_gets_no_sqlite_option(self) -> None:
        database = Database(url="postgresql+asyncpg://user:pw@host/db")
        assert database._connect_args(database.url) == {}


class TestConcurrentSchemaCreation:
    """The bug that killed the gateway on the first full four-process start.

    `create_all` reflects the existing tables and then issues CREATEs, and
    those are not one transaction. Four processes share this database and
    systemd starts them together, so two routinely reflect an empty schema
    and both try to create it — and the loser used to die with "table
    market_snapshots already exists" while the console came up fine, so
    webhooks were dead and the console looked healthy.
    """

    async def test_four_simultaneous_creators_all_succeed(self, tmp_path) -> None:
        import asyncio

        path = tmp_path / "shared.sqlite"
        databases = [Database.sqlite(path) for _ in range(4)]
        try:
            # Gathered, not awaited in turn: sequential calls cannot lose
            # the race and so cannot reproduce the bug.
            await asyncio.gather(*(db.create_schema() for db in databases))
            assert await databases[0]._schema_present() is True
        finally:
            for db in databases:
                await db.aclose()

    async def test_creating_twice_over_is_a_no_op(self, tmp_path) -> None:
        path = tmp_path / "twice.sqlite"
        database = Database.sqlite(path)
        try:
            await database.create_schema()
            await database.create_schema()
            assert await database._schema_present() is True
        finally:
            await database.aclose()

    def test_only_duplicate_errors_are_tolerated(self) -> None:
        """A bare except would turn a genuinely broken migration into a
        silent start."""
        from index_option_brain.database.engine import _is_duplicate_object

        assert _is_duplicate_object(Exception("table foo already exists")) is True
        assert _is_duplicate_object(Exception("DuplicateTable: relation exists")) is True
        assert _is_duplicate_object(Exception("disk I/O error")) is False
        assert _is_duplicate_object(Exception("no such column: spot")) is False

    async def test_an_incomplete_schema_is_not_reported_present(
        self, tmp_path
    ) -> None:
        """`_schema_present` is what stops a duplicate error from being
        read as success when the schema is in fact half-built."""
        path = tmp_path / "partial.sqlite"
        database = Database.sqlite(path)
        try:
            assert await database._schema_present() is False
            await database.create_schema()
            assert await database._schema_present() is True
        finally:
            await database.aclose()
