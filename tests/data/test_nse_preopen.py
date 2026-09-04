"""Tests for the pre-open constituent adapter.

Two properties matter more than the parsing: the weights are on the scale
the Constituent brain expects, and a board that has gone stale stops being
served. Both fail silently when wrong — the first turns a 0.35% move into
0.0035%, the second puts a 09:07 snapshot behind a 14:00 decision.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from index_option_brain.data.adapters.base import DataAdapterError
from index_option_brain.data.adapters.nse_preopen import (
    NsePreOpenAdapter,
    PreOpenSummary,
    pre_open_key,
)

BOARD_MOMENT = datetime(2026, 9, 3, 3, 37, 30, tzinfo=UTC)  # 09:07:30 IST


def row(symbol: str, iep: float, previous: float, cap: float, qty: int = 1000):
    return {
        "metadata": {
            "symbol": symbol,
            "iep": iep,
            "lastPrice": iep,
            "previousClose": previous,
            "marketCap": cap,
            "finalQuantity": qty,
        }
    }


PAYLOAD: dict[str, Any] = {
    "timestamp": "03-Sep-2026 09:07:30",
    "advances": 2,
    "declines": 1,
    "unchanged": 0,
    "totalTradedValue": 12345.6,
    "niftyPreopenStatus": {
        "lastPrice": "23997.95",
        "change": "83.5",
        "pChange": "0.35",
    },
    "data": [
        # 60% of cap, +1.00%  -> +0.60 index points of contribution
        row("HEAVY", 101.0, 100.0, 6_000.0),
        # 30% of cap, +2.00%
        row("MID", 102.0, 100.0, 3_000.0),
        # 10% of cap, -1.00%
        row("LIGHT", 99.0, 100.0, 1_000.0),
    ],
}


class _Source:
    """Stands in for the NsePublicAdapter this composes."""

    def __init__(self, payload: Any = None) -> None:
        self.payload = PAYLOAD if payload is None else payload
        self.calls = 0

    async def _get_json(self, url: str, params: Any = None) -> Any:
        self.calls += 1
        return self.payload


def at(moment: datetime):
    return lambda: moment


def test_the_endpoint_key_is_resolved_not_guessed() -> None:
    assert pre_open_key("NIFTY") == "NIFTY"
    assert pre_open_key("nifty bank") == "BANKNIFTY"
    with pytest.raises(DataAdapterError, match="no pre-open board"):
        pre_open_key("FINNIFTY")


async def test_weights_are_percentage_points_summing_to_a_hundred() -> None:
    """`ConstituentSpec.weight` is 9.82 for a 9.82% name, not 0.0982.

    The Constituent brain multiplies weight by change and divides by 100, so
    fractions here scale every contribution down by 100 — a +0.35% index move
    renders as +0.0035%, which reads as a flat market rather than a bug.
    """
    adapter = NsePreOpenAdapter(_Source(), now=at(BOARD_MOMENT))
    specs = await adapter.get_constituents("NIFTY")

    assert sum(s.weight for s in specs) == pytest.approx(Decimal(100))
    assert next(s.weight for s in specs if s.symbol == "HEAVY") == pytest.approx(
        Decimal(60)
    )


async def test_the_reconstructed_move_matches_the_exchanges_own_figure() -> None:
    """The check the whole adapter rests on.

    NSE publishes the indicative index open, and the constituent rows should
    reproduce it. If the weighting were wrong — total cap instead of free
    float, say — these two would diverge.
    """
    adapter = NsePreOpenAdapter(_Source(), now=at(BOARD_MOMENT))
    specs = await adapter.get_constituents("NIFTY")
    quotes = await adapter.get_constituent_quotes([s.symbol for s in specs])
    weights = {s.symbol: float(s.weight) for s in specs}

    reconstructed = sum(
        weights[q.symbol] * float(q.change_pct) / 100.0 for q in quotes
    )
    # 0.6*1.0 + 0.3*2.0 + 0.1*(-1.0)
    assert reconstructed == pytest.approx(1.1)


async def test_an_auction_clears_at_one_price() -> None:
    adapter = NsePreOpenAdapter(_Source(), now=at(BOARD_MOMENT))
    quote = (await adapter.get_constituent_quotes(["HEAVY"]))[0]

    assert quote.open == quote.high == quote.low == quote.ltp == Decimal("101.0")
    assert quote.timestamp == BOARD_MOMENT
    assert quote.change_pct == pytest.approx(Decimal(1))


async def test_a_stale_board_serves_nothing() -> None:
    """The snapshot never changes after the auction matches. Serving it at
    14:00 would put a confident breadth reading behind a decision that
    nothing measured."""
    adapter = NsePreOpenAdapter(
        _Source(),
        max_staleness=timedelta(minutes=45),
        now=at(BOARD_MOMENT + timedelta(hours=4)),
    )
    assert await adapter.get_constituent_quotes(["HEAVY", "MID"]) == []


async def test_a_fresh_board_serves_everything() -> None:
    adapter = NsePreOpenAdapter(
        _Source(),
        max_staleness=timedelta(minutes=45),
        now=at(BOARD_MOMENT + timedelta(minutes=20)),
    )
    assert len(await adapter.get_constituent_quotes(["HEAVY", "MID", "LIGHT"])) == 3


async def test_the_summary_is_served_even_when_stale() -> None:
    """A caller asking about the auction is asking about the auction; the
    timestamp on it says when that was."""
    adapter = NsePreOpenAdapter(
        _Source(), now=at(BOARD_MOMENT + timedelta(hours=6))
    )
    summary = await adapter.get_pre_open_summary("NIFTY")

    assert isinstance(summary, PreOpenSummary)
    assert summary.indicative_open == Decimal("23997.95")
    assert summary.advances == 2
    assert summary.advance_decline_ratio == pytest.approx(2.0)


def test_an_undefined_advance_decline_ratio_is_none_not_infinity() -> None:
    summary = PreOpenSummary(
        timestamp=BOARD_MOMENT,
        advances=50,
        declines=0,
        unchanged=0,
        indicative_open=None,
        indicative_change=None,
        indicative_change_pct=None,
        total_traded_value=None,
    )
    assert summary.advance_decline_ratio is None


async def test_an_unmatched_constituent_is_skipped_not_zeroed() -> None:
    """A name with no auction price has no price. A zero would read as a
    100% decline and drag the whole breadth reading with it."""
    payload = dict(PAYLOAD)
    payload["data"] = [*PAYLOAD["data"], row("NOMATCH", 0.0, 100.0, 500.0)]
    adapter = NsePreOpenAdapter(_Source(payload), now=at(BOARD_MOMENT))

    quotes = await adapter.get_constituent_quotes(
        ["HEAVY", "MID", "LIGHT", "NOMATCH"]
    )
    assert {q.symbol for q in quotes} == {"HEAVY", "MID", "LIGHT"}


async def test_sectors_are_reported_unknown_rather_than_guessed() -> None:
    """The payload carries no sector. A symbol-derived guess would produce
    confident rotation analysis of a partition that does not exist."""
    adapter = NsePreOpenAdapter(_Source(), now=at(BOARD_MOMENT))
    assert {s.sector for s in await adapter.get_constituents("NIFTY")} == {"UNKNOWN"}


async def test_an_empty_board_is_refused() -> None:
    adapter = NsePreOpenAdapter(_Source({"timestamp": "03-Sep-2026 09:07:30", "data": []}))
    with pytest.raises(DataAdapterError, match="empty pre-open board"):
        await adapter.get_constituents("NIFTY")
