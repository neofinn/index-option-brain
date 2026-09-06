"""The queue between a two-second webhook and a minute-long analysis."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from index_option_brain.integrations.tradingview.alert import (
    AlertRejected,
    RejectionReason,
    alert_from_payload,
)
from index_option_brain.integrations.tradingview.inbox import AlertInbox

NOW = datetime(2026, 9, 6, 4, 15, 30, tzinfo=UTC)


def alert(**overrides: object):  # type: ignore[no-untyped-def]
    payload: dict[str, object] = {
        "kind": "BREAKOUT",
        "ticker": "NIFTY",
        "interval": "5",
        "price": 24025.65,
        "bar_time": "2026-09-06T04:10:00Z",
        "fired_at": "2026-09-06T04:15:00Z",
    }
    payload.update(overrides)
    return alert_from_payload(payload)


class TestDuplicates:
    def test_the_same_bar_is_admitted_once(self) -> None:
        inbox = AlertInbox(clock=lambda: NOW)
        inbox.admit(alert())
        with pytest.raises(AlertRejected) as raised:
            inbox.admit(alert(fired_at="2026-09-06T04:15:20Z"))
        assert raised.value.reason is RejectionReason.DUPLICATE
        assert len(inbox) == 1

    def test_a_later_bar_is_a_new_alert(self) -> None:
        inbox = AlertInbox(clock=lambda: NOW)
        inbox.admit(alert())
        inbox.admit(alert(bar_time="2026-09-06T04:15:00Z"))
        assert len(inbox) == 2

    def test_draining_does_not_forget_what_was_seen(self) -> None:
        inbox = AlertInbox(clock=lambda: NOW)
        inbox.admit(alert())
        inbox.drain()
        with pytest.raises(AlertRejected):
            inbox.admit(alert())

    def test_dedupe_memory_expires_so_it_cannot_grow_without_bound(self) -> None:
        clock = [NOW]
        inbox = AlertInbox(dedupe_retention=timedelta(minutes=5), clock=lambda: clock[0])
        inbox.admit(alert())
        clock[0] = NOW + timedelta(minutes=6)
        inbox.admit(alert())  # the key aged out; no exception
        assert len(inbox) == 2


class TestBound:
    def test_the_oldest_is_dropped_and_the_drop_is_counted(self) -> None:
        """In a market the newest observation is the one worth keeping —
        but a drop that nobody counts makes a receiver outrunning the
        engine look like a healthy one."""
        inbox = AlertInbox(capacity=2, clock=lambda: NOW)
        for minute in range(3):
            inbox.admit(alert(bar_time=f"2026-09-06T04:{10 + minute:02d}:00Z"))
        assert len(inbox) == 2
        assert inbox.dropped == 1
        assert inbox.drain()[0].payload["bar_time"] == "2026-09-06T04:11:00+00:00"

    def test_a_capacity_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            AlertInbox(capacity=0)


class TestDrain:
    def test_it_returns_oldest_first_and_empties(self) -> None:
        inbox = AlertInbox(clock=lambda: NOW)
        inbox.admit(alert(bar_time="2026-09-06T04:10:00Z"))
        inbox.admit(alert(bar_time="2026-09-06T04:11:00Z"))
        drained = inbox.drain()
        assert [e.payload["bar_time"] for e in drained] == [
            "2026-09-06T04:10:00+00:00",
            "2026-09-06T04:11:00+00:00",
        ]
        assert len(inbox) == 0

    def test_peek_leaves_the_queue_alone(self) -> None:
        inbox = AlertInbox(clock=lambda: NOW)
        inbox.admit(alert())
        assert len(inbox.peek()) == 1
        assert len(inbox) == 1

    def test_stats_report_the_bound_and_the_losses(self) -> None:
        inbox = AlertInbox(capacity=4, clock=lambda: NOW)
        inbox.admit(alert())
        assert inbox.stats() == {
            "queued": 1,
            "capacity": 4,
            "dropped": 0,
            "duplicates": 0,
        }
