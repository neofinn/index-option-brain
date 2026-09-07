"""Run the gateway: `python -m index_option_brain.integrations.webhooks`.

Its own process on its own port, publicly reachable, holding no engine.

Refusing to start is a feature. With no registry file, or a registry with
no endpoints, the process exits rather than listening: a gateway with
nothing configured is a public port that answers 404 to everything, which
looks like a working deployment and is not one.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

import uvicorn

from index_option_brain.config.settings import Settings, get_settings
from index_option_brain.database.engine import Database, sqlite_url
from index_option_brain.integrations.tradingview.alert import (
    AlertRejected,
    alert_from_payload,
)
from index_option_brain.integrations.tradingview.inbox import AlertInbox
from index_option_brain.integrations.tradingview.sink import DatabaseAlertSink
from index_option_brain.integrations.webhooks.endpoints import Endpoint, load
from index_option_brain.integrations.webhooks.gateway import create_gateway_app
from index_option_brain.integrations.webhooks.store import DeliveryStore

logger = logging.getLogger(__name__)


def tradingview_handler(
    inbox: AlertInbox, sink: DatabaseAlertSink
) -> Any:
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

    database = Database(url=settings.database_url or sqlite_url(settings.sqlite_path))
    asyncio.run(database.create_schema())

    app = create_gateway_app(
        endpoints,
        DeliveryStore(database),
        on_delivery=tradingview_handler(AlertInbox(), DatabaseAlertSink(database)),
    )

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
    logger.info(
        "gateway listening on 0.0.0.0:%s", settings.webhook_gateway_port
    )
    uvicorn.run(app, host="0.0.0.0", port=settings.webhook_gateway_port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
