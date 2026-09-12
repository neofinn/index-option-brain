"""Where a strategy signal is allowed to go, and how far.

This is the file that decides whether a webhook can move money, so every
value in it is chosen by an operator editing a file on the machine — never
by the sender, never by a pushed commit, never from the web page.

The sender picks *what*, the operator picks *where*
--------------------------------------------------
The reason forwarding was left out of the gateway is that an inbound
endpoint which POSTs to any URL its caller names is an open relay:
whoever holds the ingest secret chooses the destination. A route fixes the
destination in configuration, so the strategy alert supplies only the
intent. There is deliberately no way to override the URL, the headers or
the symbol from the payload.

Off unless switched on, on the box
----------------------------------
`enabled` defaults to False and a disabled route still does everything
except the final send: it validates, maps the symbol, applies the guards,
renders exactly what it *would* have transmitted and records it. So a dry
run is a full rehearsal that produces an auditable artefact, rather than a
mode where nothing is exercised until the day it matters.

The guards are ceilings, not a strategy
---------------------------------------
`max_quantity`, `allowed_actions`, `allowed_symbols` and
`max_orders_per_day` exist to bound a strategy that has gone wrong — a
loop that fires every bar, a template with a stray zero, a chart someone
switched to the wrong symbol. They are not position sizing. This relay
does not size and must not be read as doing so: it forwards a size the
strategy chose, inside limits the operator set.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from index_option_brain.signals.contract import SignalAction

#: How stale a signal may be and still be sent onward. TradingView does
#: not retry, so a late delivery is a network fault or a replay — and
#: acting on a five-minute-old entry is how a relay buys the top of a move
#: that already finished.
DEFAULT_MAX_AGE = timedelta(seconds=90)

#: Orders per strategy per day before the route stops sending. Low on
#: purpose: the failure this catches is a strategy firing in a loop, and
#: the cost of a cap that is too low is one manual raise.
DEFAULT_MAX_ORDERS_PER_DAY = 20


@dataclass(frozen=True)
class HttpDestination:
    """A broker's REST endpoint, described entirely in configuration.

    `body_template` values are rendered with a **fixed set of named
    fields** from the signal — no expressions, no attribute access, no
    eval. A template language here would be a way to make this process
    compute something, and the whole point of the file is that it does not.
    """

    url: str
    headers: dict[str, str] = field(default_factory=dict)
    body_template: dict[str, Any] = field(default_factory=dict)
    #: The body for an `exit`, when the broker needs a different one — and
    #: it almost always does. Rendering `body_template` for an exit
    #: produces `transactionType: "EXIT"` with `quantity: 0`, which is not
    #: an order any broker accepts: closing a position means buying or
    #: selling what is actually held, and this relay does not query the
    #: account so it cannot know what that is. Without this template an
    #: exit on an HTTP destination is refused rather than sent as
    #: something that looks like an order and is not.
    #:
    #: Two shapes work. A dedicated square-off endpoint, if the broker has
    #: one; or a template using `{side_bs}`/`{action_upper}` where the
    #: *strategy* sends an explicit closing buy or sell instead of "exit".
    exit_body_template: dict[str, Any] = field(default_factory=dict)
    exit_url: str = ""
    timeout_seconds: float = 5.0
    name: str = "http"

    def __post_init__(self) -> None:
        if not self.url.startswith(("http://", "https://")):
            raise ValueError(f"destination url must be http(s): {self.url!r}")
        if self.url.startswith("http://") and not self._is_local():
            # The headers carry a broker token. Plain http to a remote host
            # puts it on the wire in clear text, and a relay is exactly the
            # component nobody looks at again after it starts working.
            raise ValueError(
                f"refusing plaintext http to a remote host: {self.url!r} — "
                "use https, or a loopback address for a local bridge"
            )
        if not self.body_template:
            raise ValueError("destination needs a body_template")
        if self.exit_url and not self.exit_url.startswith(("http://", "https://")):
            raise ValueError(f"destination exit_url must be http(s): {self.exit_url!r}")

    def _is_local(self) -> bool:
        host = self.url.split("://", 1)[1].split("/", 1)[0].split(":", 1)[0]
        return host in {"localhost", "127.0.0.1", "::1", "[::1]"}

    @property
    def handles_exit(self) -> bool:
        return bool(self.exit_body_template)

    def template_for(self, is_exit: bool) -> dict[str, Any]:
        return self.exit_body_template if is_exit else self.body_template

    def url_for(self, is_exit: bool) -> str:
        return self.exit_url if (is_exit and self.exit_url) else self.url


@dataclass(frozen=True)
class PullDestination:
    """No outbound call: the consumer polls.

    This is what an MT4/MT5 Expert Advisor needs. An EA cannot be POSTed
    to — it has no listening socket — so it reads `WebRequest` on a timer.
    The route still exists, still validates and still applies its guards;
    "sending" means the signal is durable and visible on the pull endpoint.
    """

    name: str = "pull"


Destination = HttpDestination | PullDestination


@dataclass(frozen=True)
class SignalRoute:
    """One strategy's path from alert to destination."""

    name: str
    destination: Destination
    #: TradingView ticker to the destination's own symbol. Exact keys, and
    #: an unmapped ticker is refused rather than passed through: a chart
    #: switched to the wrong symbol would otherwise send a real order for
    #: an instrument nobody meant.
    symbol_map: dict[str, str] = field(default_factory=dict)
    enabled: bool = False
    allowed_actions: frozenset[SignalAction] = frozenset(SignalAction)
    max_quantity: Decimal = Decimal(1)
    max_orders_per_day: int = DEFAULT_MAX_ORDERS_PER_DAY
    max_age: timedelta = DEFAULT_MAX_AGE
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a route needs a name")
        if not self.symbol_map:
            raise ValueError(
                f"{self.name}: symbol_map is required — an unmapped ticker "
                "must be refused, not passed through"
            )
        if self.max_quantity <= 0:
            raise ValueError(f"{self.name}: max_quantity must be positive")
        if self.max_orders_per_day < 1:
            raise ValueError(f"{self.name}: max_orders_per_day must be at least 1")
        if not self.allowed_actions:
            raise ValueError(f"{self.name}: allowed_actions cannot be empty")

    def symbol_for(self, ticker: str) -> str | None:
        return self.symbol_map.get(ticker.strip().upper()) or self.symbol_map.get(
            ticker.strip()
        )


def _destination_from(name: str, spec: dict[str, Any]) -> Destination:
    kind = str(spec.get("kind", "pull")).lower()
    if kind == "pull":
        return PullDestination()
    if kind != "http":
        raise ValueError(f"{name}: unknown destination kind {kind!r}")
    headers = {
        str(k): os.path.expandvars(str(v))
        for k, v in (spec.get("headers") or {}).items()
    }
    return HttpDestination(
        url=os.path.expandvars(str(spec.get("url", ""))),
        headers=headers,
        body_template=dict(spec.get("body_template") or {}),
        exit_body_template=dict(spec.get("exit_body_template") or {}),
        exit_url=os.path.expandvars(str(spec.get("exit_url", ""))),
        timeout_seconds=float(spec.get("timeout_seconds", 5.0)),
    )


def _quantity(value: Any, name: str) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name}: max_quantity is not a number") from exc


def load(path: str | Path) -> dict[str, SignalRoute]:
    """Read the route table from a JSON file.

    Mode 600 or tighter, enforced rather than warned about: the file holds
    broker tokens *and* the switch that turns a webhook into an order.
    """
    file = Path(path)
    if not file.exists():
        return {}
    if file.stat().st_mode & 0o077:
        raise PermissionError(
            f"{file} is readable or writable beyond its owner (mode "
            f"{oct(file.stat().st_mode & 0o777)}); it holds broker "
            "credentials and the live-trading switch — chmod 600 it"
        )
    document = json.loads(file.read_text())
    if not isinstance(document, dict):
        raise ValueError(  # noqa: TRY004 - a config value, not an argument type
            f"{file} must contain a JSON object of routes"
        )

    routes: dict[str, SignalRoute] = {}
    for name, spec in document.items():
        if name.startswith("_"):
            # Reserved for comments in the shipped example, so a copied
            # file works without the operator deleting anything.
            continue
        if not isinstance(spec, dict):
            raise ValueError(  # noqa: TRY004 - a config value
                f"{file}: route {name!r} must be an object"
            )
        actions = spec.get("allowed_actions")
        routes[name] = SignalRoute(
            name=name,
            destination=_destination_from(name, spec.get("destination") or {}),
            symbol_map={
                str(k).upper(): str(v) for k, v in (spec.get("symbol_map") or {}).items()
            },
            enabled=bool(spec.get("enabled", False)),
            allowed_actions=(
                frozenset(SignalAction(str(a).lower()) for a in actions)
                if actions
                else frozenset(SignalAction)
            ),
            max_quantity=_quantity(spec.get("max_quantity", 1), name),
            max_orders_per_day=int(
                spec.get("max_orders_per_day", DEFAULT_MAX_ORDERS_PER_DAY)
            ),
            max_age=timedelta(
                seconds=float(spec.get("max_age_seconds", DEFAULT_MAX_AGE.total_seconds()))
            ),
            description=str(spec.get("description", "")),
        )
    return routes
