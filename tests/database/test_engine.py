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
