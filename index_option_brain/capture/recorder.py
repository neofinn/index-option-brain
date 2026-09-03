"""Turn uptime into backtest corpus.

The replay engine can measure signal quality and cannot measure P&L, for one
reason: nothing free serves historical Indian option chains, so a strike
cannot be priced at a past moment. That gap is not closeable by buying data
cheaply and it *is* closeable by writing down what the system already sees.
Every session captured is a session of P&L backtesting that becomes possible
later; every session missed is gone.

So this is deliberately the most conservative component in the system:

* It writes and never reads back into a decision. Nothing here can influence
  an analysis cycle, so a bug here cannot cost a trade.
* It fails safe (spec §29). A capture failure is recorded as a system event
  and swallowed. The alternative — a full disk stopping the Execution Gate
  from gating — is strictly worse.
* It is idempotent on the feed's own timestamp, so a restart mid-session
  re-records nothing.

Cadence
-------
Two rates, because the two things being captured have different value per
byte. The market state is small and worth having densely — it is what a
session's shape is reconstructed from. The chain is ~170 rows per capture
and its value is in covering *many days at a comparable time*, not in
covering one day densely: a backtest needs a decision point per session,
not one every 20 seconds. `chain_interval` defaults to 5 minutes, which over
a year is a corpus of about 19,000 chain snapshots.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from index_option_brain.brain.pipeline import BrainCycleResult
from index_option_brain.contracts.enums import MarketSessionState
from index_option_brain.database.engine import Database
from index_option_brain.database.repository import SnapshotRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CaptureConfig:
    state_interval: timedelta = timedelta(seconds=60)
    """Minimum gap between recorded market states."""
    chain_interval: timedelta = timedelta(minutes=5)
    """Minimum gap between recorded option chains — the expensive rows."""
    capture_when_closed: bool = False
    """Whether to keep recording outside market hours.

    Off by default. A closed market republishes the same snapshot
    indefinitely, and while the dedup key makes those writes harmless they
    make the corpus look denser than the information in it.
    """
    prune_keep_days: int = 400
    """Age at which option legs are dropped. Their snapshots are kept."""


@dataclass
class CaptureStats:
    """What the capture has actually managed to do. Surfaced to the console.

    `failures` is here because a capture that quietly stopped working looks
    exactly like a capture that has nothing to do, and one of those costs a
    year of corpus.
    """

    states_written: int = 0
    chains_written: int = 0
    duplicates_skipped: int = 0
    failures: int = 0
    last_state_at: datetime | None = None
    last_chain_at: datetime | None = None
    last_error: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "states_written": self.states_written,
            "chains_written": self.chains_written,
            "duplicates_skipped": self.duplicates_skipped,
            "failures": self.failures,
            "last_state_at": (
                self.last_state_at.isoformat() if self.last_state_at else None
            ),
            "last_chain_at": (
                self.last_chain_at.isoformat() if self.last_chain_at else None
            ),
            "last_error": self.last_error,
        }


@dataclass
class CaptureRecorder:
    """Records analysis cycles to the database, on a rate limit.

    Given a cycle it decides whether this moment is worth storing, writes
    what is, and returns. It never raises: the caller is the live analysis
    loop.
    """

    database: Database
    config: CaptureConfig = field(default_factory=CaptureConfig)
    stats: CaptureStats = field(default_factory=CaptureStats)
    _ready: bool = False

    async def ensure_ready(self) -> bool:
        """Create the schema once. False means capture is unavailable."""
        if self._ready:
            return True
        try:
            await self.database.create_schema()
        except Exception as exc:
            self.stats.failures += 1
            self.stats.last_error = f"schema: {exc}"
            logger.exception("Capture schema unavailable; not recording")
            return False
        self._ready = True
        return True

    def _due(self, moment: datetime, last: datetime | None, gap: timedelta) -> bool:
        return last is None or (moment - last) >= gap

    async def record(self, result: BrainCycleResult) -> bool:
        """Record one cycle if due. Returns whether anything was written."""
        if not await self.ensure_ready():
            return False

        state = result.state
        moment = state.timestamp
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)

        if (
            not self.config.capture_when_closed
            and state.session_state is not MarketSessionState.ACTIVE
        ):
            return False
        if not self._due(moment, self.stats.last_state_at, self.config.state_interval):
            return False

        want_chain = bool(state.options_state.chain) and self._due(
            moment, self.stats.last_chain_at, self.config.chain_interval
        )

        try:
            async with self.database.session() as session:
                repository = SnapshotRepository(session)
                existing = await repository.find_snapshot(
                    state.index_symbol, moment
                )
                if existing is not None:
                    # The feed has not moved on. Nothing to add, and the
                    # timestamps are deliberately not advanced so the next
                    # genuinely new moment is still due.
                    self.stats.duplicates_skipped += 1
                    return False

                snapshot = await repository.record_snapshot(
                    state, with_chain=want_chain
                )
                await repository.record_cycle(result, snapshot=snapshot)
        except Exception as exc:
            self.stats.failures += 1
            self.stats.last_error = str(exc)
            logger.exception("Capture write failed; analysis continues")
            return False

        self.stats.states_written += 1
        self.stats.last_state_at = moment
        if want_chain:
            self.stats.chains_written += 1
            self.stats.last_chain_at = moment
        return True

    async def coverage(self, index_symbol: str) -> dict[str, object]:
        """How much corpus exists. Empty mapping when unavailable."""
        if not await self.ensure_ready():
            return {}
        try:
            async with self.database.session() as session:
                return await SnapshotRepository(session).chain_coverage(index_symbol)
        except Exception:
            logger.exception("Capture coverage unavailable")
            return {}

    async def prune(self) -> int:
        """Drop expired option legs. Returns rows removed, 0 on failure."""
        if not await self.ensure_ready():
            return 0
        try:
            async with self.database.session() as session:
                removed = await SnapshotRepository(session).prune_chains(
                    keep_days=self.config.prune_keep_days
                )
                if removed:
                    await SnapshotRepository(session).record_event(
                        "capture.prune",
                        f"Dropped {removed} option legs past "
                        f"{self.config.prune_keep_days} days",
                    )
                return removed
        except Exception:
            logger.exception("Capture prune failed")
            return 0
