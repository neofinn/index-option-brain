"""Persist observed bars across restarts.

The problem this solves
-----------------------
NSE serves no history, so bars exist only because the aggregator observed
them. That makes them expensive: a week of 5-minute bars is a week of
uptime, and until now a restart threw all of it away. On an always-on
deployment that is the difference between a system that accumulates
structure and one that is permanently blind — and it made every deploy a
decision to go blind for a day.

Why a file and not the database
-------------------------------
The §27 Postgres schema is still to be built, and the bars are needed before
it exists. A JSON file per symbol and interval is enough for the actual
requirement — survive a restart — and it has one property Postgres would not
give for free here: it works with no service running, which matters on a
single box where the point is that the thing stays up.

What it will not do
-------------------
Trust a file it cannot read. A corrupt or truncated snapshot is discarded
rather than partially loaded: half a bar series is not a shorter bar series,
it is a wrong one, and an indicator reading it would produce a confident
answer from an arbitrary window. The aggregator then starts cold, which is
recoverable; a silently mis-seeded series is not.

Writes are atomic — a temporary file renamed into place — because the process
is killed at arbitrary moments and a snapshot half-written during a restart
is exactly the file that would be loaded next.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from index_option_brain.contracts.enums import BarInterval
from index_option_brain.contracts.instruments import Bar

SCHEMA_VERSION = 1
"""Bumped when the on-disk shape changes.

A snapshot from an older version is discarded rather than migrated, for the
same reason a corrupt one is: the cost of starting cold is a day of bars, and
the cost of mis-reading a series is a wrong trade.
"""


class BarStore:
    """Bars on disk, one file per symbol and interval."""

    def __init__(self, directory: Path | str) -> None:
        self._directory = Path(directory)

    @property
    def directory(self) -> Path:
        return self._directory

    def path_for(self, symbol: str, interval: BarInterval) -> Path:
        return self._directory / f"{symbol.upper()}-{interval}.json"

    def save(self, symbol: str, interval: BarInterval, bars: list[Bar]) -> Path:
        """Write a snapshot atomically.

        Atomic because the process is killed at arbitrary moments, and a file
        half-written during a shutdown is precisely the one that would be read
        on the next start.
        """
        self._directory.mkdir(parents=True, exist_ok=True)
        target = self.path_for(symbol, interval)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "symbol": symbol.upper(),
            "interval": str(interval),
            "saved_at": datetime.now(UTC).isoformat(),
            "bars": [
                {
                    "timestamp": bar.timestamp.isoformat(),
                    "open": str(bar.open),
                    "high": str(bar.high),
                    "low": str(bar.low),
                    "close": str(bar.close),
                    "volume": bar.volume,
                }
                for bar in bars
            ],
        }
        # mkstemp rather than a context manager: the file has to survive the
        # close so it can be renamed into place, which is what makes the write
        # atomic.
        descriptor, temporary = tempfile.mkstemp(
            dir=self._directory, prefix=f".{target.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "w") as stream:
                json.dump(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        return target

    def load(self, symbol: str, interval: BarInterval) -> list[Bar]:
        """Read a snapshot, or return nothing.

        Every failure path returns an empty list rather than raising or
        partially loading. Starting cold costs a session of bars; a series
        seeded from a truncated file is a wrong series that no downstream
        indicator can detect.
        """
        path = self.path_for(symbol, interval)
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError):
            return []
        if not isinstance(payload, dict):
            return []
        if payload.get("schema_version") != SCHEMA_VERSION:
            return []
        if payload.get("interval") != str(interval):
            # A file named for one interval containing another would seed
            # 5-minute bars into a daily series.
            return []
        raw = payload.get("bars")
        if not isinstance(raw, list):
            return []

        bars: list[Bar] = []
        for entry in raw:
            if not isinstance(entry, dict):
                return []
            try:
                bars.append(
                    Bar(
                        timestamp=datetime.fromisoformat(entry["timestamp"]),
                        open=Decimal(entry["open"]),
                        high=Decimal(entry["high"]),
                        low=Decimal(entry["low"]),
                        close=Decimal(entry["close"]),
                        volume=int(entry.get("volume", 0)),
                    )
                )
            except (KeyError, TypeError, ValueError, ArithmeticError):
                # One bad bar condemns the file. A gap in the middle of a
                # series is worse than no series, because it is invisible.
                return []

        if any(bar.timestamp.tzinfo is None for bar in bars):
            # A naive timestamp would make every session boundary computed
            # from it depend on the host's timezone.
            return []
        bars.sort(key=lambda bar: bar.timestamp)
        return bars

    def saved_at(self, symbol: str, interval: BarInterval) -> datetime | None:
        """When the snapshot was written, for reporting staleness."""
        try:
            payload = json.loads(self.path_for(symbol, interval).read_text())
            return datetime.fromisoformat(payload["saved_at"])
        except (OSError, ValueError, KeyError, TypeError):
            return None
