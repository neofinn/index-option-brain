"""Tests for the NSE daily index archive adapter.

The behaviour under test is mostly about *refusal*: the adapter's value is
that a day it could not read is never mistaken for a day the market was
closed, because a silently short series is indistinguishable from a complete
one once it reaches an indicator.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

import pytest

from index_option_brain.contracts.enums import BarInterval
from index_option_brain.data.adapters.base import DataAdapterError
from index_option_brain.data.adapters.nse_archive import (
    NseArchiveAdapter,
    archive_index_name,
    archive_url,
    parse_archive_csv,
)

HEADER = (
    "Index Name,Index Date,Open Index Value,High Index Value,Low Index Value,"
    "Closing Index Value,Points Change,Change(%),Volume,Turnover (Rs. Cr.),P/E,P/B,Div Yield"
)
ROW = "Nifty 50,02-09-2026,23858,23914.45,23786.8,23914.45,-141.35,-.59,318881826,25868.81,20.22,2.89,1.19"
BODY = f"{HEADER}\n{ROW}\nNifty Next 50,02-09-2026,72319,72947.45,71821.6,72947.45,72.7,.1,211833397,10433.84,19.28,3.25,.99\n"


class _Response:
    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


class _Session:
    """Serves a scripted status per URL and counts attempts."""

    def __init__(self, script: dict[str, list[_Response]]) -> None:
        self.script = script
        self.calls: list[str] = []

    async def get(self, url: str, **_: Any) -> _Response:
        self.calls.append(url)
        queue = self.script.get(url)
        if not queue:
            return _Response(404)
        return queue.pop(0) if len(queue) > 1 else queue[0]

    async def post(self, *a: Any, **k: Any) -> _Response:  # pragma: no cover
        raise NotImplementedError

    async def delete(self, *a: Any, **k: Any) -> _Response:  # pragma: no cover
        raise NotImplementedError

    async def aclose(self) -> None:
        return None


def test_url_uses_the_archives_ddmmyyyy_form() -> None:
    assert archive_url(date(2026, 9, 2)).endswith("ind_close_all_02092026.csv")


def test_index_names_are_translated_to_the_archives_spelling() -> None:
    # The live API says "NIFTY"; the archive file says "Nifty 50".
    assert archive_index_name("NIFTY") == "nifty 50"
    assert archive_index_name("BANKNIFTY") == "nifty bank"


def test_an_unknown_index_is_refused_rather_than_guessed() -> None:
    with pytest.raises(DataAdapterError, match="No NSE archive index name"):
        archive_index_name("NIFTY IT")


def test_parsing_picks_the_requested_index_out_of_the_whole_market_file() -> None:
    bar = parse_archive_csv(BODY, index_name="nifty 50", day=date(2026, 9, 2))
    assert bar is not None
    assert bar.open == Decimal(23858)
    assert bar.high == Decimal("23914.45")
    assert bar.low == Decimal("23786.8")
    assert bar.close == Decimal("23914.45")
    assert bar.volume == 318881826


def test_a_file_without_the_index_yields_no_bar_rather_than_a_wrong_one() -> None:
    assert parse_archive_csv(BODY, index_name="nifty bank", day=date(2026, 9, 2)) is None


async def test_intraday_intervals_are_refused_not_approximated() -> None:
    adapter = NseArchiveAdapter(_Session({}))
    with pytest.raises(DataAdapterError, match="daily bars only"):
        await adapter.get_index_bars("NIFTY", BarInterval.MINUTE_5, 10)


async def test_a_404_is_a_non_trading_day_not_a_failure() -> None:
    # Only one day in the window has a file; the rest 404. That is what a
    # window spanning a long holiday looks like, and it must succeed.
    session = _Session({archive_url(date(2026, 9, 2)): [_Response(200, BODY)]})
    adapter = NseArchiveAdapter(session, max_unresolved=0)
    bars = await adapter.get_index_bars("NIFTY", BarInterval.DAY, 5, as_of=date(2026, 9, 3))
    assert len(bars) == 1
    assert bars[0].close == Decimal("23914.45")


async def test_a_transport_failure_is_retried_before_being_given_up_on() -> None:
    url = archive_url(date(2026, 9, 2))
    session = _Session({url: [_Response(503), _Response(200, BODY)]})
    adapter = NseArchiveAdapter(session, backoff=0.0, max_unresolved=0)
    bars = await adapter.get_index_bars("NIFTY", BarInterval.DAY, 5, as_of=date(2026, 9, 3))
    assert len(bars) == 1
    assert session.calls.count(url) == 2


async def test_unreadable_days_inside_the_window_refuse_the_series() -> None:
    """The bug this adapter was rewritten for.

    Two sessions publish; one is permanently unreadable. Returning the two
    good bars would hand an indicator a series with a hole in it that looks
    exactly like a complete two-bar series.
    """
    script = {
        archive_url(date(2026, 9, 2)): [_Response(200, BODY)],
        archive_url(date(2026, 9, 1)): [_Response(200, BODY)],
        archive_url(date(2026, 8, 31)): [_Response(500)],
    }
    adapter = NseArchiveAdapter(_Session(script), backoff=0.0, max_unresolved=0)
    with pytest.raises(DataAdapterError, match="unreadable"):
        await adapter.get_index_bars("NIFTY", BarInterval.DAY, 5, as_of=date(2026, 9, 3))


async def test_no_readable_session_at_all_is_an_error_not_an_empty_series() -> None:
    adapter = NseArchiveAdapter(_Session({}), backoff=0.0)
    with pytest.raises(DataAdapterError, match="no usable session"):
        await adapter.get_index_bars("NIFTY", BarInterval.DAY, 5, as_of=date(2026, 9, 3))


async def test_bars_come_back_oldest_first() -> None:
    script = {
        archive_url(date(2026, 9, 2)): [_Response(200, BODY)],
        archive_url(date(2026, 9, 1)): [_Response(200, BODY)],
    }
    adapter = NseArchiveAdapter(_Session(script), backoff=0.0, max_unresolved=0)
    bars = await adapter.get_index_bars("NIFTY", BarInterval.DAY, 5, as_of=date(2026, 9, 3))
    assert [b.timestamp.date() for b in bars] == [date(2026, 9, 1), date(2026, 9, 2)]


class TestMultiIndexFetch:
    """One archive file holds every NSE index, so a caller that wants two
    series must not fetch each day twice — and must get series that agree
    about which sessions existed."""

    BODY = (
        f"{HEADER}\n{ROW}\n"
        "India VIX,02-09-2026,11.4925,12.1125,10.5475,11.59,0.1,.87,-,-,-,-,-\n"
    )

    async def test_two_series_come_from_one_pass_over_the_files(self) -> None:
        url = archive_url(date(2026, 9, 2))
        session = _Session({url: [_Response(200, self.BODY)]})
        adapter = NseArchiveAdapter(session, max_unresolved=0)

        series = await adapter.get_many_index_bars(
            ["NIFTY", "INDIA VIX"], count=3, as_of=date(2026, 9, 3)
        )
        assert set(series) == {"NIFTY", "INDIA VIX"}
        assert series["NIFTY"][0].close == Decimal("23914.45")
        assert series["INDIA VIX"][0].close == Decimal("11.59")
        # The day was read once, not once per index.
        assert session.calls.count(url) == 1

    async def test_the_series_cover_the_same_sessions(self) -> None:
        """Misaligned series would read one session's volatility into
        another's decision, and that looks like signal."""
        script = {
            archive_url(date(2026, 9, 2)): [_Response(200, self.BODY)],
            archive_url(date(2026, 9, 1)): [_Response(200, self.BODY)],
        }
        adapter = NseArchiveAdapter(_Session(script), max_unresolved=0)
        series = await adapter.get_many_index_bars(
            ["NIFTY", "INDIA VIX"], count=5, as_of=date(2026, 9, 3)
        )
        assert [b.timestamp for b in series["NIFTY"]] == [
            b.timestamp for b in series["INDIA VIX"]
        ]

    async def test_an_index_the_archive_never_carried_is_absent(self) -> None:
        """Absent, not present and empty — the caller can then tell the
        difference between 'no history' and 'no such index here'."""
        body = f"{HEADER}\n{ROW}\n"  # NIFTY only, no VIX row
        session = _Session({archive_url(date(2026, 9, 2)): [_Response(200, body)]})
        adapter = NseArchiveAdapter(session, max_unresolved=0)

        series = await adapter.get_many_index_bars(
            ["NIFTY", "INDIA VIX"], count=3, as_of=date(2026, 9, 3)
        )
        assert "NIFTY" in series
        assert "INDIA VIX" not in series

    async def test_india_vix_resolves_to_the_archives_spelling(self) -> None:
        assert archive_index_name("INDIA VIX") == "india vix"
        assert archive_index_name("indiavix") == "india vix"
