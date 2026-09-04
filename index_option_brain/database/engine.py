"""Async engine and session plumbing for spec §21 persistence.

Two backends, one schema. PostgreSQL is the production target; SQLite is
what a freshly provisioned box has before anyone installs a database server,
and it is what the tests run against. The distinction matters because the
chain corpus in `option_snapshots` cannot be back-filled — a system that
refuses to persist until Postgres exists spends its first weeks discarding
the one thing it cannot buy later.

Failing safe (spec §29)
-----------------------
Persistence is not on the critical path of a decision. The brain runs, the
console renders and the Execution Gate gates whether or not a database is
reachable; a write failure is recorded and swallowed rather than raised into
an analysis cycle. The inverse — letting a full disk stop the system from
gating a trade — would be the more dangerous failure.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from index_option_brain.database.models import Base

logger = logging.getLogger(__name__)

#: Where a SQLite database lands when nothing else is configured. Kept beside
#: the bar snapshots so one directory is the whole persistent footprint.
DEFAULT_SQLITE_PATH = Path("var/index_brain.sqlite")


def _redact(url: str) -> str:
    """A URL safe to log: credentials removed, host and database kept.

    A connection string carries a password, and a warning about an
    unreachable database is exactly the line that ends up pasted into an
    issue.
    """
    if "@" not in url:
        return url
    scheme, _, rest = url.partition("://")
    _, _, host = rest.rpartition("@")
    return f"{scheme}://***@{host}"


def normalise_url(url: str) -> str:
    """Force an async driver onto a URL written the synchronous way.

    `postgresql://` and `sqlite:///` are what people paste from a hosting
    dashboard or a tutorial, and both fail at connect time with an error
    about greenlets that says nothing about the cause.
    """
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://", 1)
    if url.startswith("sqlite://") and "+aiosqlite" not in url:
        return url.replace("sqlite://", "sqlite+aiosqlite://", 1)
    return url


def sqlite_url(path: Path | str = DEFAULT_SQLITE_PATH) -> str:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite+aiosqlite:///{resolved}"


@dataclass
class Database:
    """An engine plus its session factory, with the schema created on demand.

    `create_all` rather than a migration on purpose, for now: there is one
    consumer, the schema is additive, and Alembic is a dependency that earns
    its place when a deployed database has data worth migrating rather than
    a corpus that can be re-captured. The models are shaped so that switch
    is mechanical.
    """

    url: str
    echo: bool = False
    _engine: AsyncEngine | None = None
    _sessions: async_sessionmaker[AsyncSession] | None = None

    @classmethod
    def sqlite(cls, path: Path | str = DEFAULT_SQLITE_PATH, **kw: object) -> Database:
        return cls(url=sqlite_url(path), **kw)  # type: ignore[arg-type]

    @classmethod
    def in_memory(cls) -> Database:
        """A private database that lives for the life of the process.

        `StaticPool` is not needed because a single AsyncEngine already
        reuses one connection for `:memory:` under aiosqlite; each Database
        instance is therefore its own isolated store, which is what a test
        wants.
        """
        return cls(url="sqlite+aiosqlite:///:memory:")

    @property
    def engine(self) -> AsyncEngine:
        if self._engine is None:
            self._engine = create_async_engine(normalise_url(self.url), echo=self.echo)
        return self._engine

    @property
    def dialect(self) -> str:
        return self.engine.dialect.name

    async def create_schema(self) -> None:
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        if self._sessions is None:
            self._sessions = async_sessionmaker(
                self.engine, expire_on_commit=False, class_=AsyncSession
            )
        return self._sessions

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """A session that commits on success and rolls back on any error."""
        async with self.session_factory()() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def is_reachable(self) -> bool:
        """Whether a connection can actually be opened and used right now.

        This exists because SQLAlchemy connects lazily: constructing a
        session succeeds against a database that is not running, and the
        failure surfaces later, inside whatever code assumed it had a
        working session. An earlier version of this class offered an
        `optional_session` context manager that yielded None "on failure" —
        it could not work, for two reasons worth recording so it is not
        reintroduced. It yielded a session that had never connected, so a
        dead database looked healthy until first use; and because the yield
        sat inside a `try`, it also swallowed exceptions raised by the
        *caller's* body and then yielded a second time, which is a
        RuntimeError from the generator machinery rather than a graceful
        degradation.

        So reachability is a question you ask, not a wrapper you hide a
        failure inside. Callers that must not fail — the capture recorder —
        own an explicit try/except around their own writes, where the scope
        of what is being tolerated is visible at the call site.
        """
        try:
            async with self.engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
        except Exception:  # noqa: BLE001 - see below
            # Deliberately blind: the question this method answers is "can I
            # use this database", and every way of failing to — bad
            # credentials, DNS, TLS, a refused socket, a driver that is not
            # installed, a read-only filesystem — is the same answer. Naming
            # a subset would let an unlisted failure propagate out of a
            # reachability *check*, which is the one place a raise is never
            # the useful outcome.
            logger.warning("Database at %s is not reachable", _redact(self.url))
            return False
        return True

    async def aclose(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._sessions = None
