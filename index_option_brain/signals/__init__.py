"""Strategy signals: an order intent from a chart, and where it may go.

`contract` normalises a TradingView strategy alert. `routes` is the
operator-owned table that decides whether it can leave the machine and how
far. `relay` is the guarded hop. `feed` is the pull side, for an Expert
Advisor that cannot be POSTed to.

This package is the one place an inbound webhook can cause an outbound
order, and a route is off until an operator enables it in a file on the
box. Read `relay`'s docstring before turning one on: nothing in this path
consults the brains, the Risk Engine or the Execution Gate.
"""

from index_option_brain.signals.contract import (
    MarketPosition,
    Signal,
    SignalAction,
    SignalRejected,
    signal_from_payload,
)
from index_option_brain.signals.feed import CSV_COLUMNS, create_signal_feed_router
from index_option_brain.signals.relay import (
    KILL_SWITCH_ENV,
    DispatchResult,
    Outcome,
    SignalRelay,
    kill_switch_engaged,
    render_body,
    template_fields,
)
from index_option_brain.signals.routes import (
    HttpDestination,
    PullDestination,
    SignalRoute,
    load,
)

__all__ = [
    "CSV_COLUMNS",
    "KILL_SWITCH_ENV",
    "DispatchResult",
    "HttpDestination",
    "MarketPosition",
    "Outcome",
    "PullDestination",
    "Signal",
    "SignalAction",
    "SignalRejected",
    "SignalRelay",
    "SignalRoute",
    "create_signal_feed_router",
    "kill_switch_engaged",
    "load",
    "render_body",
    "signal_from_payload",
    "template_fields",
]
