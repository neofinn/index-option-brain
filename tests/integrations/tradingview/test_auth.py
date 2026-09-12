"""The trust model over an endpoint that anybody can POST to."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from index_option_brain.integrations.tradingview.alert import AlertRejected, RejectionReason
from index_option_brain.integrations.tradingview.auth import (
    TRADINGVIEW_EGRESS_IPS,
    WebhookGuard,
    WebhookGuardConfig,
)

SECRET = "a-sufficiently-long-secret"
NOW = datetime(2026, 9, 6, 4, 15, 30, tzinfo=UTC)
TV_IP = next(iter(sorted(TRADINGVIEW_EGRESS_IPS)))


def body(**overrides: object) -> bytes:
    payload: dict[str, object] = {
        "secret": SECRET,
        "kind": "BREAKOUT",
        "ticker": "NIFTY",
        "interval": "5",
        "price": 24025.65,
        "bar_time": "2026-09-06T04:10:00Z",
        "fired_at": "2026-09-06T04:15:00Z",
    }
    payload.update(overrides)
    return json.dumps(payload).encode()


def guard(**config: object) -> WebhookGuard:
    settings = {"secret": SECRET, **config}
    return WebhookGuard(WebhookGuardConfig(**settings), clock=lambda: NOW)  # type: ignore[arg-type]


class TestSecret:
    def test_a_short_secret_is_refused_at_construction(self) -> None:
        """The URL appears in the alert configuration and in any
        screenshot of it. The body secret is the credential, so a
        guessable one is the whole authentication gone."""
        with pytest.raises(ValueError, match="16 characters"):
            WebhookGuardConfig(secret="short")

    def test_a_matching_secret_passes_and_is_stripped(self) -> None:
        payload = guard().authenticate(body())
        assert "secret" not in payload

    def test_a_wrong_secret_is_refused(self) -> None:
        with pytest.raises(AlertRejected) as raised:
            guard().authenticate(body(secret="wrong-but-long-enough"))
        assert raised.value.reason is RejectionReason.BAD_SECRET

    def test_a_missing_secret_is_refused(self) -> None:
        payload = json.loads(body())
        del payload["secret"]
        with pytest.raises(AlertRejected) as raised:
            guard().authenticate(json.dumps(payload).encode())
        assert raised.value.reason is RejectionReason.BAD_SECRET


class TestBody:
    def test_a_non_json_body_is_refused(self) -> None:
        with pytest.raises(AlertRejected) as raised:
            guard().authenticate(b"BUY NIFTY NOW")
        assert raised.value.reason is RejectionReason.MALFORMED_BODY

    def test_a_json_array_is_not_an_alert(self) -> None:
        with pytest.raises(AlertRejected) as raised:
            guard().authenticate(b'["breakout"]')
        assert raised.value.reason is RejectionReason.MALFORMED_BODY

    def test_an_oversized_body_is_refused_before_it_is_parsed(self) -> None:
        with pytest.raises(AlertRejected) as raised:
            guard().authenticate(b"x" * 9000)
        assert raised.value.reason is RejectionReason.BODY_TOO_LARGE


class TestSourceAddress:
    def test_a_tradingview_address_passes(self) -> None:
        guard().check_source(TV_IP)

    def test_another_address_is_refused(self) -> None:
        with pytest.raises(AlertRejected) as raised:
            guard().check_source("203.0.113.9")
        assert raised.value.reason is RejectionReason.SOURCE_NOT_ALLOWED

    def test_an_empty_allowlist_turns_the_check_off(self) -> None:
        """A real deployment behind a tunnel, where the peer address is
        the tunnel. It has to be chosen, which is why the default is the
        published set and not an empty one."""
        guard(allowed_ips=frozenset()).check_source("203.0.113.9")

    def test_a_forwarded_header_is_ignored_by_default(self) -> None:
        """X-Forwarded-For is client-supplied text. Honouring it by
        default would let any sender name their own address and reduce
        the allowlist to decoration."""
        source = guard().source_address("203.0.113.9", {"x-forwarded-for": TV_IP})
        assert source == "203.0.113.9"

    def test_a_forwarded_header_is_honoured_when_a_proxy_is_declared(self) -> None:
        source = guard(trust_forwarded_for=True).source_address(
            "10.0.0.2", {"x-forwarded-for": f"{TV_IP}, 10.0.0.1"}
        )
        assert source == TV_IP


class TestFreshness:
    def test_a_fresh_alert_passes(self) -> None:
        guard().admit(body(), peer_ip=TV_IP)

    def test_a_stale_alert_is_refused(self) -> None:
        """TradingView does not retry, so a late delivery is a fault or a
        replay — and acting on a ten-minute-old breakout is worse than
        dropping it."""
        with pytest.raises(AlertRejected) as raised:
            guard().admit(body(fired_at="2026-09-06T04:00:00Z"), peer_ip=TV_IP)
        assert raised.value.reason is RejectionReason.STALE

    def test_a_future_dated_alert_is_refused(self) -> None:
        with pytest.raises(AlertRejected) as raised:
            guard().admit(body(fired_at="2026-09-06T05:15:00Z"), peer_ip=TV_IP)
        assert raised.value.reason is RejectionReason.FUTURE_DATED

    def test_small_clock_skew_is_tolerated(self) -> None:
        guard().admit(body(fired_at="2026-09-06T04:15:45Z"), peer_ip=TV_IP)

    def test_the_window_is_configurable(self) -> None:
        guard(max_age=timedelta(hours=1)).admit(
            body(fired_at="2026-09-06T04:00:00Z"), peer_ip=TV_IP
        )


class TestCheckOrder:
    def test_a_wrong_address_never_learns_whether_its_secret_was_close(self) -> None:
        """Source is checked before the secret, so probing from a
        disallowed address is uninformative about the credential."""
        with pytest.raises(AlertRejected) as raised:
            guard().admit(body(secret="wrong-but-long-enough"), peer_ip="203.0.113.9")
        assert raised.value.reason is RejectionReason.SOURCE_NOT_ALLOWED


class TestStats:
    def test_outcomes_are_counted_so_a_silent_reject_loop_is_visible(self) -> None:
        """A receiver dropping everything looks, from the chart, exactly
        like one nobody has fired at."""
        g = guard()
        g.admit(body(), peer_ip=TV_IP)
        with pytest.raises(AlertRejected):
            g.admit(body(secret="wrong-but-long-enough"), peer_ip=TV_IP)
        assert g.stats.as_dict() == {
            "accepted": 1,
            "rejected": {"BAD_SECRET": 1},
        }
