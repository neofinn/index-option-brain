"""Run the webhook receiver: `python -m index_option_brain.integrations.tradingview`.

Its own process, its own port. The console is served on the tailnet and
must stay there; this one answers TradingView's servers on the public
internet, and the two have opposite exposure requirements.

Refusing to start is a feature
------------------------------
With no `TRADINGVIEW_WEBHOOK_SECRET` the process exits instead of starting
with a default. An open POST endpoint on the public internet that anyone
who finds the URL can push market claims into is worse than no receiver at
all, and a placeholder secret is exactly how that happens.
"""

from __future__ import annotations

import asyncio
import logging
import sys

import uvicorn

from index_option_brain.config.settings import Settings, get_settings
from index_option_brain.database.engine import Database, sqlite_url
from index_option_brain.integrations.tradingview.auth import (
    TRADINGVIEW_EGRESS_IPS,
    WebhookGuard,
    WebhookGuardConfig,
)
from index_option_brain.integrations.tradingview.inbox import AlertInbox
from index_option_brain.integrations.tradingview.receiver import create_webhook_app
from index_option_brain.integrations.tradingview.sink import DatabaseAlertSink

logger = logging.getLogger(__name__)


def allowed_ips(setting: str) -> frozenset[str]:
    """Parse `TRADINGVIEW_ALLOWED_IPS`.

    Empty means TradingView's published egress set. The literal `any` is
    the explicit opt-out for a tunnelled deployment where the peer address
    is the tunnel rather than TradingView.
    """
    value = setting.strip()
    if not value:
        return TRADINGVIEW_EGRESS_IPS
    if value.lower() == "any":
        return frozenset()
    return frozenset(part.strip() for part in value.split(",") if part.strip())


def build_config(settings: Settings) -> WebhookGuardConfig:
    return WebhookGuardConfig(
        secret=settings.tradingview_webhook_secret,
        allowed_ips=allowed_ips(settings.tradingview_allowed_ips),
        trust_forwarded_for=settings.tradingview_trust_forwarded_for,
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = get_settings()

    if not settings.tradingview_webhook_secret:
        logger.error(
            "TRADINGVIEW_WEBHOOK_SECRET is not set; refusing to start an "
            "unauthenticated public endpoint"
        )
        return 2
    try:
        config = build_config(settings)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2

    database = Database(url=settings.database_url or sqlite_url(settings.sqlite_path))
    asyncio.run(database.create_schema())

    app = create_webhook_app(
        WebhookGuard(config),
        AlertInbox(),
        DatabaseAlertSink(database),
    )
    if not config.allowed_ips:
        logger.warning(
            "the source-address check is off; the shared secret is the only "
            "thing standing between this endpoint and the internet"
        )
    logger.info("TradingView receiver listening on 0.0.0.0:%s", settings.tradingview_webhook_port)
    uvicorn.run(app, host="0.0.0.0", port=settings.tradingview_webhook_port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
