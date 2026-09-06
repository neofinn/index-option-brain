"""The webhook receiver: a separate application, on a separate port.

Why not a route on the console API
----------------------------------
The console application has a test asserting that every route it exposes is
GET, HEAD or OPTIONS. That is not tidiness — it is the boundary the
assistant sits behind. Clawdbot can reach the console, and "the console
cannot be made to do anything" is a property proved by there being no verb
that could. Adding the first POST there would delete that proof for every
route at once.

There is a second, blunter reason. A TradingView webhook must be reachable
from the public internet, because TradingView's servers are the ones
calling. The console is served on the tailnet precisely so that it is not.
Merging them would drag the console onto the public internet to satisfy the
webhook.

So this is its own ASGI app, its own process, its own port: publicly
exposed, guarded, and unable to reach a broker because it holds no
reference to one.

What it can and cannot do
-------------------------
It holds three objects: a `WebhookGuard`, an `AlertInbox` and an
`AlertSink`. It can authenticate a request, put an `Event` in a queue, and
write that event to the operational log. It has no engine, no order
manager, no risk engine, and no settings object that could hand it one —
and there is a test that walks this package's import graph to keep it that
way. An alert cannot place an order here because there is nothing here to
place one with.

It is also silent by design. The only readable route is a liveness probe
that returns a constant. Queue depth, rejection counts and the events
themselves are read in-process by the console, over the tailnet, and are
not exposed on the address TradingView posts to — a public endpoint that
reports how many alerts fired and what they said tells anyone who finds the
URL what the system is watching.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from index_option_brain.integrations.tradingview.alert import AlertRejected, RejectionReason
from index_option_brain.integrations.tradingview.auth import WebhookGuard
from index_option_brain.integrations.tradingview.inbox import AlertInbox
from index_option_brain.integrations.tradingview.sink import AlertSink, NullAlertSink

logger = logging.getLogger(__name__)

#: Rejections that must not say why. Distinguishing "wrong secret" from
#: "unknown symbol" in the response tells someone probing the endpoint when
#: they have guessed the credential; every other reason is a diagnostic the
#: operator needs in TradingView's alert log and is safe once the secret has
#: already matched.
_OPAQUE: frozenset[RejectionReason] = frozenset(
    {RejectionReason.BAD_SECRET, RejectionReason.SOURCE_NOT_ALLOWED}
)

_STATUS: dict[RejectionReason, int] = {
    RejectionReason.BAD_SECRET: 401,
    RejectionReason.SOURCE_NOT_ALLOWED: 401,
    RejectionReason.BODY_TOO_LARGE: 413,
    RejectionReason.MALFORMED_BODY: 422,
    RejectionReason.MISSING_FIELD: 422,
    RejectionReason.BAD_FIELD: 422,
    RejectionReason.UNKNOWN_SYMBOL: 422,
    RejectionReason.NOT_CHART_OBSERVABLE: 422,
    RejectionReason.STALE: 409,
    RejectionReason.FUTURE_DATED: 409,
}


def _rejection_response(rejected: AlertRejected) -> JSONResponse:
    if rejected.reason in _OPAQUE:
        return JSONResponse({"ok": False}, status_code=401)
    body: dict[str, Any] = {"ok": False, "reason": str(rejected.reason)}
    if rejected.detail:
        body["detail"] = rejected.detail
    return JSONResponse(body, status_code=_STATUS.get(rejected.reason, 422))


def create_webhook_app(
    guard: WebhookGuard, inbox: AlertInbox, sink: AlertSink | None = None
) -> FastAPI:
    """The receiver, over one guard, one inbox and one sink, and nothing else.

    `sink` defaults to `NullAlertSink`, which keeps the alert in memory and
    nowhere else. That is correct only when the consumer holds this same
    inbox; the two-process deployment needs `DatabaseAlertSink`.
    """

    writer = sink or NullAlertSink()

    app = FastAPI(
        title="TradingView alert receiver",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.get("/tv/health")
    async def health() -> dict[str, str]:
        """A constant. Deliberately carries no counts: this route answers to
        the public internet, and queue depth is information about what the
        system is watching."""
        return {"status": "ok"}

    @app.post("/tv/webhook")
    async def webhook(request: Request) -> JSONResponse:
        peer = request.client.host if request.client else None
        raw = await request.body()
        try:
            alert = guard.admit(raw, peer_ip=peer, headers=request.headers)
        except AlertRejected as rejected:
            return _rejection_response(rejected)

        try:
            event = inbox.admit(alert)
        except AlertRejected as rejected:
            # A redelivery of a bar already accepted is not a failure of the
            # alert. Answering 4xx here would make TradingView's log show an
            # error for a webhook that worked exactly as intended.
            guard.stats.record_rejection(rejected.reason)
            guard.stats.accepted -= 1
            return JSONResponse(
                {"ok": True, "accepted": False, "reason": str(rejected.reason)},
                status_code=200,
            )

        try:
            await writer.record(event)
        except Exception:
            # Answering 200 here would tell TradingView's alert log that
            # everything worked while nothing downstream will ever see the
            # alert. A failed webhook in that log is the signal.
            logger.exception("failed to record a TradingView alert")
            return JSONResponse(
                {"ok": False, "reason": "NOT_PERSISTED"},
                status_code=503,
            )

        return JSONResponse({"ok": True, "accepted": True, "event_id": event.event_id})

    return app
