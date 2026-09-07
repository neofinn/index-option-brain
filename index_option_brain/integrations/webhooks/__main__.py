"""Run the gateway: `python -m index_option_brain.integrations.webhooks`.

Its own process on its own port, publicly reachable, holding no engine.

Refusing to start is a feature. With no registry file, or a registry with
no endpoints, the process exits rather than listening: a gateway with
nothing configured is a public port that answers 404 to everything, which
looks like a working deployment and is not one.

This is the one module in the package that wires in the signal relay, and
therefore the one place where an inbound delivery can cause an outbound
order. `gateway.py` itself stays free of it — it takes a generic delivery
handler — so the import-graph test still holds for the service, and the
capability lives in exactly one file that can be read end to end.

Every route is off until its own `enabled` is set in the routes file, and
`SIGNAL_RELAY_KILL=1` stops all of them regardless. The startup log says
which routes would actually send, because a relay that is live and a relay
that is rehearsing look identical from the outside.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import uvicorn

from index_option_brain.config.settings import Settings, get_settings
from index_option_brain.data.http import HttpxSession
from index_option_brain.database.engine import Database, sqlite_url
from index_option_brain.integrations.tradingview.alert import (
    AlertRejected,
    alert_from_payload,
)
from index_option_brain.integrations.tradingview.inbox import AlertInbox
from index_option_brain.integrations.tradingview.sink import DatabaseAlertSink
from index_option_brain.integrations.webhooks.endpoints import (
    Endpoint,
    EndpointKind,
    load,
)
from index_option_brain.integrations.webhooks.gateway import create_gateway_app
from index_option_brain.integrations.webhooks.store import DeliveryStore
from index_option_brain.signals import routes as signal_routes
from index_option_brain.signals.feed import create_signal_feed_router
from index_option_brain.signals.relay import (
    KILL_SWITCH_ENV,
    SignalRelay,
    kill_switch_engaged,
)

logger = logging.getLogger(__name__)

#: What the gateway hands a specialised endpoint kind, and what it expects
#: back: a note for the sender, or None. Named because two factories below
#: produce one and mypy needs the return type to be more than `Any`.
DeliveryHandler = Callable[[Endpoint, dict[str, Any]], Awaitable[str | None]]


def tradingview_handler(
    inbox: AlertInbox, sink: DatabaseAlertSink
) -> DeliveryHandler:
    """The strict alert path, as a gateway delivery handler.

    The gateway has already authenticated the delivery, stripped the
    credential and stored the payload — so a chart alert that fails
    validation here is still readable over the API. What this adds is the
    claim check: an alert may only assert a trigger a chart could actually
    observe, and one claiming an IV collapse is refused rather than turned
    into an event.

    The reason string is returned rather than raised, because it reaches
    TradingView's own alert log that way. A rejection only a server log
    knows about is one the operator finds out about days later.
    """

    async def handle(endpoint: Endpoint, payload: dict[str, Any]) -> str | None:
        try:
            alert = alert_from_payload(payload)
            event = inbox.admit(alert)
        except AlertRejected as rejected:
            logger.info("%s: alert not accepted — %s", endpoint.slug, rejected)
            return str(rejected.reason)
        await sink.record(event)
        return "accepted"

    return handle


def delivery_handler(
    inbox: AlertInbox, sink: DatabaseAlertSink, relay: SignalRelay
) -> DeliveryHandler:
    """One handler over both specialised endpoint kinds.

    `tradingview` deliveries become `Event`s for the engine to reason
    about; `strategy` deliveries go to the relay, which may forward them.
    Branching here rather than inside either path keeps the difference
    between "wakes the brain" and "can place an order" visible in one
    place.
    """
    tradingview = tradingview_handler(inbox, sink)

    async def handle(endpoint: Endpoint, payload: dict[str, Any]) -> str | None:
        if endpoint.kind is EndpointKind.STRATEGY:
            result = await relay.dispatch(endpoint.slug, payload)
            if result.reason:
                logger.info(
                    "%s: %s — %s", endpoint.slug, result.outcome, result.reason
                )
            # The outcome reaches TradingView's alert log, so a blocked
            # signal is visible where the operator is already looking.
            return (
                result.outcome
                if not result.reason
                else f"{result.outcome}: {result.reason}"
            )
        return await tradingview(endpoint, payload)

    return handle


def registry_path(settings: Settings) -> Path:
    return Path(settings.webhook_endpoints_file)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = get_settings()
    path = registry_path(settings)

    try:
        endpoints = load(path)
    except (ValueError, PermissionError, OSError) as exc:
        logger.error("%s", exc)
        return 2
    if not endpoints:
        logger.error(
            "no endpoints configured in %s; refusing to start a gateway that "
            "would answer 404 to everything while looking healthy",
            path,
        )
        return 2

    try:
        routes = signal_routes.load(Path(settings.signal_routes_file))
    except (ValueError, PermissionError, OSError) as exc:
        logger.error("%s", exc)
        return 2

    strategy_slugs = {
        slug for slug, ep in endpoints.items() if ep.kind is EndpointKind.STRATEGY
    }
    missing = sorted(strategy_slugs - set(routes))
    if missing:
        # A strategy endpoint with no route accepts orders and silently
        # discards them. Starting anyway would give the operator a webhook
        # that answers 200 and does nothing, which is worse than a refusal.
        logger.error(
            "strategy endpoints with no route in %s: %s",
            settings.signal_routes_file,
            ", ".join(missing),
        )
        return 2

    database = Database(url=settings.database_url or sqlite_url(settings.sqlite_path))
    asyncio.run(database.create_schema())

    relay = SignalRelay(routes, database, session=HttpxSession())
    app = create_gateway_app(
        endpoints,
        DeliveryStore(database),
        on_delivery=delivery_handler(
            AlertInbox(), DatabaseAlertSink(database), relay
        ),
    )
    app.include_router(create_signal_feed_router(relay, endpoints))

    for endpoint in sorted(endpoints.values(), key=lambda e: e.slug):
        logger.info(
            "POST /hook/%s  (%s, keeps %d, %s)",
            endpoint.slug,
            endpoint.kind,
            endpoint.retain,
            f"{len(endpoint.allowed_ips)} allowed IPs"
            if endpoint.allowed_ips
            else "any source address",
        )
    if kill_switch_engaged():
        logger.warning(
            "%s is set: no route will send, whatever its config says",
            KILL_SWITCH_ENV,
        )
    for route in sorted(routes.values(), key=lambda r: r.name):
        # Said at every start, for both states. A relay that is live and
        # one that is rehearsing are indistinguishable from the outside,
        # and the log is the only place the difference is stated.
        logger.warning(
            "route %s -> %s: %s (max qty %s, %d/day, symbols %s)",
            route.name,
            route.destination.name,
            "LIVE — deliveries will be sent" if route.enabled else "dry run",
            route.max_quantity,
            route.max_orders_per_day,
            ",".join(sorted(route.symbol_map)),
        )
    logger.info(
        "gateway listening on 0.0.0.0:%s", settings.webhook_gateway_port
    )
    uvicorn.run(app, host="0.0.0.0", port=settings.webhook_gateway_port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
