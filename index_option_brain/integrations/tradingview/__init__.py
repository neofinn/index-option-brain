"""TradingView as an external detector.

A chart alert enters this system the same way an internal detector's finding
does — as an `Event` meaning "something changed; analyze it" — and gets no
more authority than that. See `receiver` for why it runs as its own
application, and `alert` for what a chart is allowed to claim.
"""

from index_option_brain.integrations.tradingview.alert import (
    CHART_OBSERVABLE,
    TICKER_TO_SYMBOL,
    AlertRejected,
    RejectionReason,
    TradingViewAlert,
    alert_from_payload,
    resolve_symbol,
)
from index_option_brain.integrations.tradingview.auth import (
    TRADINGVIEW_EGRESS_IPS,
    GuardStats,
    WebhookGuard,
    WebhookGuardConfig,
)
from index_option_brain.integrations.tradingview.inbox import AlertInbox
from index_option_brain.integrations.tradingview.receiver import create_webhook_app
from index_option_brain.integrations.tradingview.sink import (
    ALERT_KIND,
    AlertSink,
    DatabaseAlertSink,
    NullAlertSink,
    pending_alerts,
)

__all__ = [
    "ALERT_KIND",
    "CHART_OBSERVABLE",
    "TICKER_TO_SYMBOL",
    "TRADINGVIEW_EGRESS_IPS",
    "AlertInbox",
    "AlertRejected",
    "AlertSink",
    "DatabaseAlertSink",
    "GuardStats",
    "NullAlertSink",
    "RejectionReason",
    "TradingViewAlert",
    "WebhookGuard",
    "WebhookGuardConfig",
    "alert_from_payload",
    "create_webhook_app",
    "pending_alerts",
    "resolve_symbol",
]
