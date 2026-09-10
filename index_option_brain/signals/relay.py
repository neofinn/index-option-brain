"""The guarded hop from a strategy alert to a broker or an EA.

This module is the one place in the system where an inbound webhook can
cause an outbound order, and it is written on the assumption that it will
one day be handed a signal that should not be sent.

What it deliberately is not
---------------------------
It is **not** the decision path. Nothing here consults the nine brains,
the Regime Engine, the Risk Engine or the Execution Gate. A signal that
arrives is a signal that goes, inside the route's ceilings — which means
turning a route on delegates the decision to the TradingView strategy and
to nothing else in this repository. That is a legitimate thing to want and
it is what was asked for; it is also the single largest reduction in
safety this codebase has, so it is stated here rather than discovered.

The engine's own path still exists and is unchanged: an alert on a
`tradingview` gateway endpoint becomes an `Event`, wakes the pipeline, and
whatever comes out still has to pass risk and the gate. A route in this
file bypasses all of it.

The five ways a signal stops here
---------------------------------
1. **Kill switch.** One environment variable, checked on every dispatch,
   ahead of everything else.
2. **Route disabled.** The default. A disabled route rehearses in full and
   records what it would have sent.
3. **Guards.** Action, symbol, quantity ceiling, staleness, daily cap.
4. **Duplicate.** The database's unique constraint, not a memory set — so
   a restart between two deliveries of one intent cannot double-order.
5. **Destination failure.** Recorded with the status and body, never
   retried automatically: a blind retry on a broker call whose response
   was lost is how one intent becomes two positions.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from index_option_brain.data.http import HttpError, HttpSession
from index_option_brain.database.engine import Database
from index_option_brain.database.models import SignalDispatchRow
from index_option_brain.signals.contract import (
    Signal,
    SignalAction,
    SignalRejected,
    signal_from_payload,
)
from index_option_brain.signals.routes import (
    HttpDestination,
    PullDestination,
    SignalRoute,
)

logger = logging.getLogger(__name__)

#: Set this to anything truthy and no route sends, whatever its config
#: says. A single variable an operator can set over SSH in one command,
#: because the moment you need it you do not want to be editing JSON.
KILL_SWITCH_ENV = "SIGNAL_RELAY_KILL"

#: A second way in, for the same switch. The environment variable cannot
#: be set by anything but the relay's own process, and the moment you most
#: need to stop trading you may be holding a phone rather than a terminal
#: — so the chat bot engages it by creating this file instead.
#:
#: Deliberately one-way from outside: the bot can create the file, and
#: only someone at the machine can remove it. A switch that can be flipped
#: back from chat is one an argument in a group chat can turn off.
KILL_SWITCH_FILE_ENV = "SIGNAL_RELAY_KILL_FILE"
DEFAULT_KILL_SWITCH_FILE = "var/RELAY_KILLED"

_TRUTHY = {"1", "true", "yes", "on"}

RESPONSE_BODY_LIMIT = 2000


class Outcome:
    CLAIMED = "CLAIMED"
    SENT = "SENT"
    DRY_RUN = "DRY_RUN"
    BLOCKED = "BLOCKED"
    DUPLICATE = "DUPLICATE"
    FAILED = "FAILED"


@dataclass(frozen=True)
class DispatchResult:
    outcome: str
    reason: str | None = None
    seq: int | None = None
    request_preview: dict[str, Any] | None = None
    response_status: int | None = None
    signal: Signal | None = None
    #: True when the payload itself was the problem — a malformed body, a
    #: chart on a symbol this route does not map. False for every policy
    #: refusal: a daily cap, a disabled route, an engaged kill switch and a
    #: staleness drop are the operator's or the network's, and reporting
    #: them to the sender as failures makes one's own guard look like a
    #: broken webhook.
    sender_error: bool = False

    @property
    def ok(self) -> bool:
        """Whether the relay did what the route asked of it.

        A dry run is `ok`: the route is configured not to send, and it did
        not send. A duplicate is `ok` too — the intent is already handled,
        which is success rather than an error the sender should retry.
        """
        return self.outcome in {Outcome.SENT, Outcome.DRY_RUN, Outcome.DUPLICATE}

    def as_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {"outcome": self.outcome, "ok": self.ok}
        if self.reason:
            body["reason"] = self.reason
        if self.seq is not None:
            body["seq"] = self.seq
        if self.response_status is not None:
            body["response_status"] = self.response_status
        return body


def kill_switch_path(environ: Mapping[str, str] | None = None) -> Path:
    env = environ if environ is not None else os.environ
    return Path(env.get(KILL_SWITCH_FILE_ENV) or DEFAULT_KILL_SWITCH_FILE)


def kill_switch_engaged(environ: Mapping[str, str] | None = None) -> bool:
    """Either way of engaging it counts.

    Checked on every dispatch rather than cached, because a cached kill
    switch is one that does not take effect until a restart — which is the
    opposite of what it is for.
    """
    env = environ if environ is not None else os.environ
    if env.get(KILL_SWITCH_ENV, "").strip().lower() in _TRUTHY:
        return True
    try:
        os.stat(kill_switch_path(env))
    except FileNotFoundError:
        # The only answer that means "definitely not engaged".
        return False
    except (OSError, ValueError):
        # Anything else — a permission error on the parent, an
        # unrepresentable path — means the file's absence cannot be
        # established, and an unestablished absence is not a reason to
        # start trading. `Path.exists()` is deliberately not used here: it
        # returns False for every one of these, which makes "no kill file"
        # and "cannot see whether there is a kill file" the same answer.
        return True
    return True


def _number(value: Decimal) -> int | float:
    """A Decimal as the JSON number a broker expects.

    Integral values become `int`, so a quantity of 2 serialises as `2` and
    not `2.0` — several broker APIs reject the latter for a lot count.
    """
    return int(value) if value == value.to_integral_value() else float(value)


def template_fields(signal: Signal, symbol: str) -> dict[str, Any]:
    """The fixed set of names a body template may reference.

    Fixed on purpose. A template language with attribute access or
    expressions would make this file compute things, and the reason it is
    configuration is that it does not.
    """
    return {
        "strategy": signal.strategy,
        "ticker": signal.ticker,
        "symbol": symbol,
        "action": str(signal.action),
        "action_upper": str(signal.action).upper(),
        # Two spellings brokers actually use, so a template does not need a
        # conditional to produce them.
        "side_bs": {"buy": "B", "sell": "S", "exit": "X"}[str(signal.action)],
        "side_long_short": {"buy": "LONG", "sell": "SHORT", "exit": "FLAT"}[
            str(signal.action)
        ],
        "quantity": Decimal(0) if signal.quantity is None else signal.quantity,
        "target": (
            Decimal(0) if signal.resolved_target is None else signal.resolved_target
        ),
        "price": Decimal(0) if signal.price is None else signal.price,
        "order_id": signal.order_id or "",
        "comment": signal.comment or "",
        "bar_time": signal.bar_time.isoformat(),
        "fired_at": signal.fired_at.isoformat(),
    }


def render_body(
    template: dict[str, Any], fields: Mapping[str, Any]
) -> dict[str, Any]:
    """Substitute the template's placeholders.

    A value that is *exactly* one placeholder over a numeric field becomes
    a JSON number; anything else is string-formatted. That rule exists
    because `{"quantity": "{quantity}"}` and `{"quantity": "lots={quantity}"}`
    want different types out, and guessing from the field name alone would
    quote a quantity that a broker then rejects.
    """

    def render(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: render(inner) for key, inner in value.items()}
        if isinstance(value, list):
            return [render(item) for item in value]
        if not isinstance(value, str):
            return value
        bare = value.strip()
        if bare.startswith("{") and bare.endswith("}") and bare.count("{") == 1:
            key = bare[1:-1]
            if key in fields:
                found = fields[key]
                return _number(found) if isinstance(found, Decimal) else found
        try:
            return value.format(**fields)
        except (KeyError, IndexError) as exc:
            raise SignalRejected(
                f"body_template references a field this relay does not "
                f"provide: {exc}"
            ) from exc

    rendered = render(template)
    assert isinstance(rendered, dict)
    return rendered


class SignalRelay:
    def __init__(
        self,
        routes: dict[str, SignalRoute],
        database: Database,
        *,
        session: HttpSession | None = None,
        clock: Callable[[], datetime] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self._routes = routes
        self._database = database
        self._session = session
        self._clock = clock or (lambda: datetime.now(UTC))
        self._environ = environ

    @property
    def routes(self) -> dict[str, SignalRoute]:
        return self._routes

    async def _sent_today(self, route: str, today: date) -> int:
        """Only SENT and DRY_RUN count against the cap.

        A blocked signal consuming a slot would let a misconfigured
        template exhaust the day's allowance without a single order
        reaching anyone — the cap is there to bound orders, not attempts.
        """
        start = datetime.combine(today, datetime.min.time(), tzinfo=UTC)
        statement = select(func.count(SignalDispatchRow.seq)).where(
            SignalDispatchRow.route == route,
            SignalDispatchRow.received_at >= start,
            SignalDispatchRow.outcome.in_([Outcome.SENT, Outcome.DRY_RUN]),
        )
        async with self._database.session() as session:
            return int((await session.execute(statement)).scalar_one() or 0)

    async def _record_blocked(
        self,
        route: SignalRoute,
        signal: Signal | None,
        payload: dict[str, Any],
        reason: str,
        *,
        sender_error: bool = False,
    ) -> DispatchResult:
        """A refusal is written down too.

        An audit trail that records only what was sent cannot answer the
        question actually asked after a bad day, which is what was refused
        and why.
        """
        now = self._clock()
        try:
            async with self._database.session() as session:
                row = SignalDispatchRow(
                    route=route.name,
                    idempotency_key=(
                        signal.idempotency_key
                        if signal
                        else f"blocked:{now.isoformat()}:{abs(hash(json.dumps(payload, sort_keys=True, default=str))) % 10**12}"
                    ),
                    strategy=(signal.strategy if signal else str(payload.get("strategy", "?"))[:64]),
                    ticker=(signal.ticker if signal else str(payload.get("ticker", "?"))[:32]),
                    symbol=None,
                    action=(str(signal.action) if signal else str(payload.get("action", "?"))[:8]),
                    quantity=signal.quantity if signal else None,
                    target_position=signal.target_position if signal else None,
                    received_at=now,
                    outcome=Outcome.BLOCKED,
                    reason=reason[:200],
                    request_preview={},
                )
                session.add(row)
        except IntegrityError:
            # A repeat of an already-blocked signal. The refusal is already
            # on record; writing it again adds nothing.
            pass
        return DispatchResult(
            outcome=Outcome.BLOCKED,
            reason=reason,
            signal=signal,
            sender_error=sender_error,
        )

    async def dispatch(self, route_name: str, payload: dict[str, Any]) -> DispatchResult:
        """Take one strategy alert as far as its route allows."""
        route = self._routes.get(route_name)
        if route is None:
            return DispatchResult(
                outcome=Outcome.BLOCKED, reason=f"no route named {route_name!r}"
            )

        if kill_switch_engaged(self._environ):
            # Ahead of everything, including parsing. When this is set the
            # answer is no, and no amount of well-formedness changes it.
            return await self._record_blocked(
                route, None, payload, f"{KILL_SWITCH_ENV} is set"
            )

        try:
            signal = signal_from_payload(payload)
        except SignalRejected as rejected:
            # The alert template is wrong. The sender has to see this as a
            # failure or they will never look at it.
            return await self._record_blocked(
                route, None, payload, rejected.reason, sender_error=True
            )

        if signal.action not in route.allowed_actions:
            return await self._record_blocked(
                route, signal, payload, f"action {signal.action} is not allowed on this route"
            )

        symbol = route.symbol_for(signal.ticker)
        if symbol is None:
            # Usually a chart left on the wrong symbol, which is the
            # sender's to fix and invisible to them unless it is reported.
            return await self._record_blocked(
                route,
                signal,
                payload,
                f"ticker {signal.ticker!r} is not in this route's symbol map",
                sender_error=True,
            )

        age = (self._clock() - signal.fired_at).total_seconds()
        if age > route.max_age.total_seconds():
            return await self._record_blocked(
                route, signal, payload, f"signal fired {int(age)}s ago"
            )
        if age < -30:
            return await self._record_blocked(
                route, signal, payload, "signal is dated in the future"
            )

        if signal.quantity is not None and signal.quantity > route.max_quantity:
            return await self._record_blocked(
                route,
                signal,
                payload,
                f"quantity {signal.quantity} exceeds the route ceiling "
                f"{route.max_quantity}",
            )
        target = signal.resolved_target
        if target is not None and abs(target) > route.max_quantity:
            return await self._record_blocked(
                route,
                signal,
                payload,
                f"target position {target} exceeds the route ceiling "
                f"{route.max_quantity}",
            )

        sent = await self._sent_today(route.name, self._clock().date())
        if sent >= route.max_orders_per_day:
            return await self._record_blocked(
                route,
                signal,
                payload,
                f"daily cap reached ({sent}/{route.max_orders_per_day})",
            )

        destination = route.destination
        if (
            isinstance(destination, HttpDestination)
            and signal.action is SignalAction.EXIT
            and not destination.handles_exit
        ):
            # Refused rather than sent. Rendering the entry template for an
            # exit produces transactionType "EXIT" with quantity 0, which
            # no broker accepts — and which a mock broker answers 200 to,
            # so it looks like it worked right up until it matters.
            return await self._record_blocked(
                route,
                signal,
                payload,
                "this HTTP destination has no exit_body_template, and an "
                "entry template cannot express closing a position",
            )

        try:
            preview = self._preview(route, signal, symbol)
        except SignalRejected as rejected:
            return await self._record_blocked(route, signal, payload, rejected.reason)

        seq = await self._claim(route, signal, symbol, preview)
        if seq is None:
            return DispatchResult(
                outcome=Outcome.DUPLICATE,
                reason="this intent has already been handled",
                signal=signal,
            )

        if not route.enabled:
            await self._finish(seq, Outcome.DRY_RUN, reason="route is not enabled")
            return DispatchResult(
                outcome=Outcome.DRY_RUN,
                reason="route is not enabled",
                seq=seq,
                request_preview=preview,
                signal=signal,
            )

        return await self._send(route, signal, symbol, seq, preview)

    def _preview(
        self, route: SignalRoute, signal: Signal, symbol: str
    ) -> dict[str, Any]:
        """What would be transmitted, credentials excluded.

        Header *names* are kept and their values are not: knowing that an
        `access-token` header was set is what an operator needs when a
        broker answers 401, and the value is the one thing that must never
        be in an audit row.
        """
        destination = route.destination
        if isinstance(destination, PullDestination):
            return {
                "destination": "pull",
                "symbol": symbol,
                "signal": signal.as_dict(),
            }
        is_exit = signal.action is SignalAction.EXIT
        body = render_body(
            destination.template_for(is_exit), template_fields(signal, symbol)
        )
        return {
            "destination": "http",
            "url": destination.url_for(is_exit),
            "header_names": sorted(destination.headers),
            "body": body,
        }

    async def _claim(
        self,
        route: SignalRoute,
        signal: Signal,
        symbol: str,
        preview: dict[str, Any],
    ) -> int | None:
        """Reserve this intent, or return None if it is already reserved.

        The unique constraint does the work, in its own transaction, before
        anything is transmitted. In memory this check would not survive the
        restart that happens between two deliveries of one alert.
        """
        try:
            async with self._database.session() as session:
                row = SignalDispatchRow(
                    route=route.name,
                    idempotency_key=signal.idempotency_key,
                    strategy=signal.strategy,
                    ticker=signal.ticker,
                    symbol=symbol,
                    action=str(signal.action),
                    quantity=signal.quantity,
                    target_position=signal.target_position,
                    received_at=self._clock(),
                    outcome=Outcome.CLAIMED,
                    request_preview=preview,
                )
                session.add(row)
                await session.flush()
                return row.seq
        except IntegrityError:
            return None

    async def _finish(
        self,
        seq: int,
        outcome: str,
        *,
        reason: str | None = None,
        status: int | None = None,
        body: str | None = None,
    ) -> None:
        async with self._database.session() as session:
            row = await session.get(SignalDispatchRow, seq)
            if row is None:
                return
            row.outcome = outcome
            row.reason = reason[:200] if reason else None
            row.sent_at = self._clock()
            row.response_status = status
            row.response_body = body[:RESPONSE_BODY_LIMIT] if body else None

    async def _send(
        self,
        route: SignalRoute,
        signal: Signal,
        symbol: str,
        seq: int,
        preview: dict[str, Any],
    ) -> DispatchResult:
        destination = route.destination
        if isinstance(destination, PullDestination):
            # Nothing to transmit: the consumer polls. The signal is
            # durable and on the pull endpoint, which is what "sent" means
            # for an EA that has no listening socket.
            await self._finish(seq, Outcome.SENT, reason="queued for pull")
            return DispatchResult(
                outcome=Outcome.SENT,
                reason="queued for pull",
                seq=seq,
                request_preview=preview,
                signal=signal,
            )

        if self._session is None:
            await self._finish(
                seq, Outcome.FAILED, reason="no HTTP session was provided to the relay"
            )
            return DispatchResult(
                outcome=Outcome.FAILED,
                reason="no HTTP session was provided to the relay",
                seq=seq,
                signal=signal,
            )

        assert isinstance(destination, HttpDestination)
        try:
            response = await self._session.post(
                # From the preview, so what was audited is exactly what is
                # transmitted — including the exit URL when there is one.
                str(preview["url"]),
                json=preview["body"],
                headers=destination.headers,
            )
        except (HttpError, OSError) as exc:
            # Not retried. A broker call whose response was lost may well
            # have been executed, and a blind retry is how one intent
            # becomes two positions.
            logger.warning("%s: destination call failed — %s", route.name, exc)
            await self._finish(seq, Outcome.FAILED, reason=str(exc)[:200])
            return DispatchResult(
                outcome=Outcome.FAILED, reason=str(exc), seq=seq, signal=signal
            )

        status = getattr(response, "status_code", None)
        text = getattr(response, "text", "") or ""
        if status is None or not (200 <= int(status) < 300):
            await self._finish(
                seq,
                Outcome.FAILED,
                reason=f"destination answered {status}",
                status=int(status) if status is not None else None,
                body=text,
            )
            return DispatchResult(
                outcome=Outcome.FAILED,
                reason=f"destination answered {status}",
                seq=seq,
                response_status=int(status) if status is not None else None,
                signal=signal,
            )

        await self._finish(seq, Outcome.SENT, status=int(status), body=text)
        logger.info(
            "%s: %s %s %s -> %s (%s)",
            route.name,
            signal.strategy,
            signal.action,
            symbol,
            destination.url,
            status,
        )
        return DispatchResult(
            outcome=Outcome.SENT,
            seq=seq,
            request_preview=preview,
            response_status=int(status),
            signal=signal,
        )

    async def recent(self, route: str, *, cursor: int = 0, limit: int = 50) -> list[SignalDispatchRow]:
        """Dispatches after `cursor`, oldest first — the EA's pull feed.

        An integer cursor for the same reason the gateway uses one: `seq`
        is unique, so `>` is exact and a poller that stores the last value
        it saw never skips or repeats.
        """
        statement = (
            select(SignalDispatchRow)
            .where(SignalDispatchRow.route == route, SignalDispatchRow.seq > cursor)
            .order_by(SignalDispatchRow.seq.asc())
            .limit(max(1, min(limit, 200)))
        )
        async with self._database.session() as session:
            return list((await session.execute(statement)).scalars().all())
