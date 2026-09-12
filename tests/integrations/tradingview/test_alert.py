"""What a chart alert is allowed to assert, and what it becomes."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from index_option_brain.contracts.enums import TriggerType
from index_option_brain.integrations.tradingview.alert import (
    CHART_OBSERVABLE,
    AlertRejected,
    RejectionReason,
    alert_from_payload,
    resolve_symbol,
)

FIRED = "2026-09-06T04:15:00Z"
BAR = "2026-09-06T04:10:00Z"


def payload(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "kind": "BREAKOUT",
        "ticker": "NIFTY",
        "interval": "5",
        "price": 24025.65,
        "bar_time": BAR,
        "fired_at": FIRED,
    }
    body.update(overrides)
    return body


class TestSymbolResolution:
    def test_it_maps_a_known_ticker(self) -> None:
        assert resolve_symbol("NSE:NIFTY") == "NIFTY"

    def test_banknifty_does_not_collapse_into_nifty(self) -> None:
        """The reason the map is exact rather than a substring rule. A
        substring match sends every BANKNIFTY alert to NIFTY with a
        plausible price attached, and nothing downstream can tell."""
        assert resolve_symbol("BANKNIFTY") == "BANKNIFTY"
        assert resolve_symbol("NSE:BANKNIFTY1!") == "BANKNIFTY"

    def test_it_is_case_and_space_insensitive(self) -> None:
        assert resolve_symbol("  nse:nifty  ") == "NIFTY"

    def test_an_unmapped_ticker_is_refused_not_guessed(self) -> None:
        with pytest.raises(AlertRejected) as raised:
            resolve_symbol("NSE:FINNIFTY")
        assert raised.value.reason is RejectionReason.UNKNOWN_SYMBOL


class TestChartObservability:
    def test_a_price_trigger_is_accepted(self) -> None:
        alert = alert_from_payload(payload())
        assert alert.trigger_type is TriggerType.BREAKOUT

    @pytest.mark.parametrize(
        "kind",
        ["IV_EXPANSION_COLLAPSE", "LARGE_OI_ADDITION", "BREADTH_CHANGE", "GAMMA_CONCENTRATION_CHANGE"],
    )
    def test_a_trigger_needing_data_a_chart_cannot_see_is_refused(self, kind: str) -> None:
        """Pine sees one symbol's OHLCV. An alert claiming an IV collapse
        is not a reading, it is a label over data the sender never had."""
        with pytest.raises(AlertRejected) as raised:
            alert_from_payload(payload(kind=kind))
        assert raised.value.reason is RejectionReason.NOT_CHART_OBSERVABLE

    def test_the_observable_set_holds_no_option_or_constituent_trigger(self) -> None:
        for trigger in CHART_OBSERVABLE:
            assert trigger in {
                TriggerType.SIGNIFICANT_PRICE_MOVEMENT,
                TriggerType.BREAKOUT,
                TriggerType.BREAKDOWN,
                TriggerType.VWAP_CROSSING,
                TriggerType.SUPPORT_RESISTANCE_TEST,
                TriggerType.OPENING_RANGE_EVENT,
                TriggerType.VOLATILITY_EXPANSION_CONTRACTION,
                TriggerType.VOLUME_ANOMALY,
            }

    def test_an_unknown_trigger_name_is_refused(self) -> None:
        with pytest.raises(AlertRejected) as raised:
            alert_from_payload(payload(kind="MOON_PHASE"))
        assert raised.value.reason is RejectionReason.BAD_FIELD


class TestFieldParsing:
    def test_a_float_price_keeps_its_decimal_digits(self) -> None:
        """Decimal(24025.65) carries binary residue; Decimal(str(...))
        does not, and a strike comparison against 24025.65 has to match."""
        assert alert_from_payload(payload(price=24025.65)).price == Decimal("24025.65")

    def test_a_quoted_price_is_accepted_too(self) -> None:
        assert alert_from_payload(payload(price="24025.65")).price == Decimal("24025.65")

    def test_a_boolean_is_not_a_price(self) -> None:
        with pytest.raises(AlertRejected) as raised:
            alert_from_payload(payload(price=True))
        assert raised.value.reason is RejectionReason.BAD_FIELD

    def test_a_naive_timestamp_is_read_as_utc(self) -> None:
        """The receiver may run in any zone. Reading a UTC instant as IST
        would put every staleness check five and a half hours out."""
        alert = alert_from_payload(payload(fired_at="2026-09-06T04:15:00"))
        assert alert.fired_at == datetime(2026, 9, 6, 4, 15, tzinfo=UTC)

    def test_an_offset_timestamp_is_converted(self) -> None:
        alert = alert_from_payload(payload(fired_at="2026-09-06T09:45:00+05:30"))
        assert alert.fired_at == datetime(2026, 9, 6, 4, 15, tzinfo=UTC)

    def test_a_missing_required_field_names_itself(self) -> None:
        body = payload()
        del body["price"]
        with pytest.raises(AlertRejected) as raised:
            alert_from_payload(body)
        assert raised.value.reason is RejectionReason.MISSING_FIELD
        assert "price" in raised.value.detail

    def test_an_absent_level_stays_absent(self) -> None:
        assert alert_from_payload(payload()).level is None

    def test_a_long_note_is_truncated_rather_than_refused(self) -> None:
        alert = alert_from_payload(payload(note="x" * 5000))
        assert alert.note is not None
        assert len(alert.note) == 280


class TestEventConversion:
    def test_two_deliveries_of_one_bar_derive_one_event_id(self) -> None:
        """Redelivery across a reconnect changes {{timenow}} but not the
        bar. Keying the id on the fire time would make one alert two
        events in every store that keys on id."""
        first = alert_from_payload(payload()).to_event()
        second = alert_from_payload(payload(fired_at="2026-09-06T04:15:09Z")).to_event()
        assert first.event_id == second.event_id

    def test_a_different_bar_is_a_different_event(self) -> None:
        first = alert_from_payload(payload()).to_event()
        second = alert_from_payload(payload(bar_time="2026-09-06T04:15:00Z")).to_event()
        assert first.event_id != second.event_id

    def test_the_event_carries_no_significance_score(self) -> None:
        """Significance is scored against this system's own market state.
        A number from the sender would be an unverified input deciding
        whether the pipeline wakes."""
        assert alert_from_payload(payload()).to_event().significance_score is None

    def test_the_payload_names_its_source(self) -> None:
        event = alert_from_payload(payload()).to_event()
        assert event.payload["source"] == "tradingview"
        assert event.payload["symbol"] == "NIFTY"

    def test_a_secret_left_in_the_body_cannot_reach_the_event(self) -> None:
        """The guard strips it, and the model has no field to hold it —
        so even a guard bug cannot put the credential into a persisted
        event."""
        alert = alert_from_payload(payload(secret="leaked-secret-value"))
        assert "secret" not in alert.model_dump()
        assert "leaked-secret-value" not in str(alert.to_event().model_dump())
