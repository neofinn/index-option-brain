"""Turning a strategy alert into an unambiguous order intent."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from index_option_brain.signals.contract import (
    MarketPosition,
    SignalAction,
    SignalRejected,
    signal_from_payload,
)


def payload(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "strategy": "orb-nifty",
        "ticker": "NIFTY",
        "action": "buy",
        "quantity": 2,
        "bar_time": "2026-09-07T04:10:00Z",
        "fired_at": "2026-09-07T04:15:00Z",
    }
    body.update(overrides)
    return body


class TestNothingIsInvented:
    def test_a_signal_with_no_size_at_all_is_refused(self) -> None:
        """The most expensive default in the system. A size nobody chose is
        a real position nobody chose."""
        body = payload()
        del body["quantity"]
        with pytest.raises(SignalRejected, match="refusing to invent a size"):
            signal_from_payload(body)

    def test_a_target_position_is_size_enough(self) -> None:
        body = payload(target_position=-1)
        del body["quantity"]
        assert signal_from_payload(body).resolved_target == Decimal(-1)

    def test_an_exit_needs_no_size(self) -> None:
        """"Close everything" is fully specified without a quantity."""
        body = payload(action="exit")
        del body["quantity"]
        assert signal_from_payload(body).resolved_target == Decimal(0)

    def test_a_zero_quantity_is_refused(self) -> None:
        with pytest.raises(SignalRejected, match="positive"):
            signal_from_payload(payload(quantity=0))

    @pytest.mark.parametrize("field", ["strategy", "ticker", "action"])
    def test_a_missing_required_field_is_refused(self, field: str) -> None:
        body = payload()
        del body[field]
        with pytest.raises(SignalRejected, match=field):
            signal_from_payload(body)


class TestTargetBeatsAction:
    def test_the_target_wins_when_both_are_sent(self) -> None:
        """Webhooks are at-most-once. A relay acting on deltas is
        permanently out of step after one lost alert; one acting on the
        target self-heals on the next signal."""
        signal = signal_from_payload(payload(quantity=1, target_position=3))
        assert signal.resolved_target == Decimal(3)

    def test_a_delta_only_signal_reports_no_target(self) -> None:
        """So the relay knows it has to ask the destination what it
        holds, rather than guessing."""
        assert signal_from_payload(payload()).resolved_target is None


class TestIdempotency:
    def test_a_refire_of_the_same_bar_is_the_same_intent(self) -> None:
        """TradingView re-fires on a reconnect and replays on a chart
        reload, both with a fresh {{timenow}}. Keying on that would place
        the order twice."""
        first = signal_from_payload(payload())
        second = signal_from_payload(payload(fired_at="2026-09-07T04:15:41Z"))
        assert first.idempotency_key == second.idempotency_key

    def test_a_different_bar_is_a_different_intent(self) -> None:
        first = signal_from_payload(payload())
        second = signal_from_payload(payload(bar_time="2026-09-07T04:15:00Z"))
        assert first.idempotency_key != second.idempotency_key

    def test_a_different_size_on_the_same_bar_is_a_different_intent(self) -> None:
        """A strategy that revises its size within one bar means it, and
        collapsing the two would drop the revision."""
        assert (
            signal_from_payload(payload(quantity=2)).idempotency_key
            != signal_from_payload(payload(quantity=3)).idempotency_key
        )


class TestParsing:
    @pytest.mark.parametrize(
        ("word", "expected"),
        [
            ("buy", SignalAction.BUY),
            ("LONG", SignalAction.BUY),
            ("sell", SignalAction.SELL),
            ("short", SignalAction.SELL),
            ("exit", SignalAction.EXIT),
            ("close", SignalAction.EXIT),
            ("flat", SignalAction.EXIT),
        ],
    )
    def test_the_words_strategies_actually_send(
        self, word: str, expected: SignalAction
    ) -> None:
        body = payload(action=word)
        if expected is SignalAction.EXIT:
            del body["quantity"]
        assert signal_from_payload(body).action is expected

    def test_an_unknown_action_is_refused(self) -> None:
        with pytest.raises(SignalRejected, match="action"):
            signal_from_payload(payload(action="hedge"))

    def test_a_quoted_number_is_accepted(self) -> None:
        assert signal_from_payload(payload(quantity="2")).quantity == Decimal(2)

    def test_a_float_keeps_its_digits(self) -> None:
        """A lot-size check on 0.35000000000000003 fails for no reason a
        reader could see."""
        assert signal_from_payload(payload(quantity=0.35)).quantity == Decimal("0.35")

    def test_a_naive_timestamp_is_read_as_utc(self) -> None:
        signal = signal_from_payload(payload(fired_at="2026-09-07T04:15:00"))
        assert signal.fired_at == datetime(2026, 9, 7, 4, 15, tzinfo=UTC)

    def test_market_position_is_parsed(self) -> None:
        assert (
            signal_from_payload(payload(market_position="Long")).market_position
            is MarketPosition.LONG
        )

    def test_an_unknown_market_position_is_refused(self) -> None:
        with pytest.raises(SignalRejected, match="market_position"):
            signal_from_payload(payload(market_position="hedged"))

    def test_a_secret_left_in_the_body_cannot_reach_the_signal(self) -> None:
        signal = signal_from_payload(payload(secret="leaked-value"))
        assert "leaked-value" not in str(signal.as_dict())
