"""Deciding whether an inbound POST is really a TradingView alert.

TradingView cannot sign a webhook. There is no HMAC header, no mutual TLS,
no per-alert nonce — the request is a bare POST that anybody who learns the
URL can reproduce. So the trust model here is built from what is actually
available, and each part is load-bearing because none of them is sufficient
alone:

**A shared secret in the body.** The only credential TradingView can carry.
Compared with `hmac.compare_digest`, and stripped before the body is parsed
so it cannot survive into an event, a log line or an HTTP response.

**A source-address allowlist.** TradingView posts from a small, published
set of egress addresses. This is what makes a leaked URL insufficient on
its own — and it is the reason `trust_forwarded_for` defaults to False. An
`X-Forwarded-For` header is client-supplied text; honouring it by default
would let any sender name their own address and reduce the allowlist to
decoration. It is enabled only when a reverse proxy the operator controls
is known to overwrite the header.

**A freshness window.** A captured request replays forever otherwise.
TradingView does not retry a failed webhook, so a delivery arriving minutes
late is not a retry — it is a network fault or a replay, and acting on a
ten-minute-old breakout is worse than dropping it. Duplicate suppression
sits one layer up in the inbox, which knows what has already been admitted.

What this deliberately is not
-----------------------------
It is not authorization to trade. Passing every check here means the sender
is probably TradingView and the alert is fresh. It does not mean the
observation is true, and nothing downstream treats it as more than a reason
to go and look at the market for itself.
"""

from __future__ import annotations

import hmac
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from index_option_brain.integrations.tradingview.alert import (
    AlertRejected,
    RejectionReason,
    TradingViewAlert,
    alert_from_payload,
)

#: TradingView's published webhook egress addresses. Held as a default
#: rather than a constant in the check, because a vendor can change these
#: and an operator must be able to correct it without a code change.
TRADINGVIEW_EGRESS_IPS: frozenset[str] = frozenset(
    {
        "52.89.214.238",
        "34.212.75.30",
        "54.218.53.128",
        "52.32.178.7",
    }
)

MAX_BODY_BYTES = 8192


@dataclass(frozen=True)
class WebhookGuardConfig:
    """How strict the receiver is, with every relaxation named.

    `allowed_ips` empty means the address check is off. That is a real
    deployment (behind Cloudflare Tunnel or ngrok the peer address is the
    tunnel, not TradingView) but it must be chosen, so the default is the
    published set rather than an empty one.
    """

    secret: str
    allowed_ips: frozenset[str] = TRADINGVIEW_EGRESS_IPS
    trust_forwarded_for: bool = False
    max_age: timedelta = timedelta(seconds=120)
    max_clock_skew: timedelta = timedelta(seconds=30)
    max_body_bytes: int = MAX_BODY_BYTES

    def __post_init__(self) -> None:
        if len(self.secret) < 16:
            # A webhook URL is not secret in practice: it appears in the
            # alert configuration, in TradingView's logs, and in any
            # screenshot of either. The body secret is the credential, so a
            # guessable one is the whole authentication gone.
            raise ValueError("the webhook secret must be at least 16 characters")


@dataclass
class GuardStats:
    """Counts by outcome, so a receiver that is rejecting everything says so.

    A webhook that silently drops every alert looks identical, from the
    chart, to one nobody has fired at — the alert log shows a 200 either
    way. These counters are what makes the difference visible on the
    console.
    """

    accepted: int = 0
    rejected: dict[str, int] = field(default_factory=dict)

    def record_rejection(self, reason: RejectionReason) -> None:
        self.rejected[str(reason)] = self.rejected.get(str(reason), 0) + 1

    def as_dict(self) -> dict[str, Any]:
        return {"accepted": self.accepted, "rejected": dict(self.rejected)}


class WebhookGuard:
    """Authenticates a raw request body and returns a validated alert."""

    def __init__(
        self,
        config: WebhookGuardConfig,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._config = config
        self._clock = clock or (lambda: datetime.now(UTC))
        self.stats = GuardStats()

    @property
    def config(self) -> WebhookGuardConfig:
        return self._config

    def source_address(self, peer_ip: str | None, headers: Mapping[str, str]) -> str | None:
        """The address to hold against the allowlist.

        With `trust_forwarded_for` off this is the peer address and nothing
        else, whatever headers claim. With it on, the *leftmost* entry of
        `X-Forwarded-For` is used: a proxy appends, so the leftmost value is
        the original client as the first trusted proxy saw it.
        """
        if not self._config.trust_forwarded_for:
            return peer_ip
        forwarded = headers.get("x-forwarded-for") or headers.get("X-Forwarded-For")
        if forwarded:
            first = forwarded.split(",")[0].strip()
            if first:
                return first
        return peer_ip

    def check_source(self, address: str | None) -> None:
        allowed = self._config.allowed_ips
        if not allowed:
            return
        if address is None or address not in allowed:
            raise AlertRejected(
                RejectionReason.SOURCE_NOT_ALLOWED,
                "the source address is not in the allowlist",
            )

    def authenticate(self, raw_body: bytes) -> dict[str, Any]:
        """Verify the shared secret and return the body without it.

        The secret is removed from the returned mapping, not merely ignored,
        so no later stage can copy it forward.
        """
        if len(raw_body) > self._config.max_body_bytes:
            raise AlertRejected(
                RejectionReason.BODY_TOO_LARGE,
                f"body exceeds {self._config.max_body_bytes} bytes",
            )
        try:
            # TradingView sends the alert message verbatim and sets
            # `Content-Type: text/plain` unless the body parses as JSON on
            # its side, so the content type is not something to route on.
            # The body is decoded and parsed here regardless of what the
            # header claims.
            payload = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AlertRejected(RejectionReason.MALFORMED_BODY, "body is not JSON") from exc
        if not isinstance(payload, dict):
            raise AlertRejected(RejectionReason.MALFORMED_BODY, "body is not a JSON object")

        supplied = payload.pop("secret", None)
        if not isinstance(supplied, str) or not hmac.compare_digest(
            supplied, self._config.secret
        ):
            raise AlertRejected(RejectionReason.BAD_SECRET, "the shared secret did not match")
        return payload

    def check_freshness(self, alert: TradingViewAlert) -> None:
        now = self._clock()
        age = now - alert.fired_at
        if age > self._config.max_age:
            raise AlertRejected(
                RejectionReason.STALE,
                f"alert fired {int(age.total_seconds())}s ago",
            )
        if -age > self._config.max_clock_skew:
            # A future-dated alert is either a clock the sender controls or
            # a fabricated body; both make the freshness window meaningless.
            raise AlertRejected(
                RejectionReason.FUTURE_DATED,
                "alert is dated in the future beyond the allowed skew",
            )

    def admit(
        self,
        raw_body: bytes,
        *,
        peer_ip: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> TradingViewAlert:
        """The whole check, in the order that leaks the least.

        Source first, then secret, then shape, then freshness — an address
        that is not TradingView's never gets to learn whether its guessed
        secret was close, and never gets its body parsed.
        """
        try:
            self.check_source(self.source_address(peer_ip, headers or {}))
            payload = self.authenticate(raw_body)
            alert = alert_from_payload(payload)
            self.check_freshness(alert)
        except AlertRejected as rejected:
            self.stats.record_rejection(rejected.reason)
            raise
        self.stats.accepted += 1
        return alert
