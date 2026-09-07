"""What a TradingView *strategy* alert says, normalised.

An indicator alert is an observation ("the range broke"). A strategy alert
is an **order intent** ("go long 2 contracts"), and it reaches this system
through the same webhook. The difference matters enough to justify a
separate contract: an observation wakes the pipeline, an intent is meant
to move money.

Relay the target, not the action
--------------------------------
TradingView gives both. `{{strategy.order.action}}` is a *delta* — buy 2,
sell 1 — and `{{strategy.position_size}}` is the **position the strategy
believes it now holds** after that order. Relaying deltas is the obvious
design and it is the wrong one: webhooks are at-most-once, so one lost
alert leaves a delta relay permanently out of step with the chart, and it
stays wrong for every trade after it. A relay that acts on the target
self-heals — the next signal restates the whole truth, and the difference
between the target and the live position is the order to send.

So `target_position` is preferred wherever it is present, and `action` is
the fallback for a strategy template that does not send it.

Nothing is invented
-------------------
A signal with no quantity and no target is **refused**, not defaulted to
one lot. This is the "absence is not zero" rule at the point where it is
most expensive: a default size is a real position nobody chose. The one
exception is `exit`, which is fully specified without a quantity — its
target is flat, and that is a fact rather than a guess.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any


class SignalRejected(Exception):
    """A strategy alert that cannot be turned into an unambiguous intent."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class SignalAction(StrEnum):
    BUY = "buy"
    SELL = "sell"
    #: Close whatever is open. Distinct from a sell: selling into a flat
    #: account opens a short, which is the opposite of what an exit means.
    EXIT = "exit"


class MarketPosition(StrEnum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


_ACTION_WORDS = {
    "buy": SignalAction.BUY,
    "long": SignalAction.BUY,
    "sell": SignalAction.SELL,
    "short": SignalAction.SELL,
    "exit": SignalAction.EXIT,
    "close": SignalAction.EXIT,
    "flat": SignalAction.EXIT,
    "flatten": SignalAction.EXIT,
}

_POSITION_WORDS = {
    "long": MarketPosition.LONG,
    "short": MarketPosition.SHORT,
    "flat": MarketPosition.FLAT,
}


def _decimal(value: Any, field: str) -> Decimal | None:
    """A number from whatever JSON type the alert template produced.

    `{{strategy.order.contracts}}` substitutes a bare number, but an
    operator who quotes it in the template sends a string, and both are
    correct. Floats route through `str()` so 0.35 stays 0.35 rather than
    acquiring binary residue that a lot-size check would then trip on.
    """
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise SignalRejected(f"{field!r} is not a number")
    try:
        if isinstance(value, float | int):
            return Decimal(str(value))
        if isinstance(value, str):
            return Decimal(value.strip())
    except (InvalidOperation, ValueError) as exc:
        raise SignalRejected(f"{field!r} is not a number") from exc
    raise SignalRejected(f"{field!r} is not a number")


def _timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise SignalRejected(f"{field!r} is required and must be a timestamp")
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError as exc:
        raise SignalRejected(f"{field!r} is not an ISO-8601 timestamp") from exc
    # Naive is read as UTC, never as local time: the relay may run in any
    # zone and reinterpreting a UTC instant as IST puts every staleness
    # check five and a half hours out.
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


@dataclass(frozen=True)
class Signal:
    """One normalised order intent from a strategy."""

    strategy: str
    ticker: str
    action: SignalAction
    bar_time: datetime
    fired_at: datetime
    quantity: Decimal | None = None
    target_position: Decimal | None = None
    market_position: MarketPosition | None = None
    order_id: str | None = None
    price: Decimal | None = None
    comment: str | None = None

    def __post_init__(self) -> None:
        if self.action is not SignalAction.EXIT:
            if self.quantity is None and self.target_position is None:
                # The most expensive default in the system. A size nobody
                # chose is a real position nobody chose.
                raise SignalRejected(
                    "neither a quantity nor a target position was sent; "
                    "refusing to invent a size"
                )
            if self.quantity is not None and self.quantity <= 0:
                raise SignalRejected("quantity must be positive")

    @property
    def resolved_target(self) -> Decimal | None:
        """The signed position this signal wants to end up holding.

        `None` when the alert sent only a delta, which is the case a relay
        has to handle by asking the destination what it currently holds.
        An exit is always a target of zero — no quantity needed, because
        "close everything" is fully specified without one.
        """
        if self.action is SignalAction.EXIT:
            return Decimal(0)
        if self.target_position is not None:
            return self.target_position
        return None

    @property
    def idempotency_key(self) -> str:
        """What makes two deliveries the same intent.

        Keyed on the strategy, the bar and the order id — never on the
        firing time. TradingView re-fires an alert on a reconnect and
        replays one on a chart reload, and both arrive with a fresh
        `{{timenow}}`. Keying on that would place the order twice.
        """
        parts = [
            self.strategy,
            self.ticker,
            str(self.action),
            self.bar_time.isoformat(),
            self.order_id or "",
            "" if self.quantity is None else str(self.quantity),
            "" if self.target_position is None else str(self.target_position),
        ]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:32]

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "ticker": self.ticker,
            "action": str(self.action),
            "quantity": None if self.quantity is None else str(self.quantity),
            "target_position": (
                None if self.target_position is None else str(self.target_position)
            ),
            "market_position": (
                None if self.market_position is None else str(self.market_position)
            ),
            "order_id": self.order_id,
            "price": None if self.price is None else str(self.price),
            "bar_time": self.bar_time.isoformat(),
            "fired_at": self.fired_at.isoformat(),
            "comment": self.comment,
            "idempotency_key": self.idempotency_key,
        }


def _action(value: Any) -> SignalAction:
    if not isinstance(value, str) or not value.strip():
        raise SignalRejected("'action' is required")
    word = value.strip().lower()
    action = _ACTION_WORDS.get(word)
    if action is None:
        raise SignalRejected(
            f"'action' {value!r} is not one of "
            f"{', '.join(sorted(set(_ACTION_WORDS)))}"
        )
    return action


def signal_from_payload(payload: dict[str, Any]) -> Signal:
    """Normalise a strategy alert body into a `Signal`.

    Called with the ingest credential already stripped by the gateway, so
    nothing here can carry it forward.

    Field names follow TradingView's own placeholders rather than being
    renamed to something tidier: an operator writing the alert template is
    reading TradingView's documentation, and a contract that matches it is
    one they can fill in without a translation table.
    """
    strategy = payload.get("strategy") or payload.get("strategy_name")
    if not isinstance(strategy, str) or not strategy.strip():
        # Which strategy fired is not decoration: it is what a per-strategy
        # daily order cap and an audit trail are keyed on.
        raise SignalRejected("'strategy' is required")

    ticker = payload.get("ticker")
    if not isinstance(ticker, str) or not ticker.strip():
        raise SignalRejected("'ticker' is required")

    raw_position = payload.get("market_position")
    market_position = None
    if isinstance(raw_position, str) and raw_position.strip():
        market_position = _POSITION_WORDS.get(raw_position.strip().lower())
        if market_position is None:
            raise SignalRejected(
                f"'market_position' {raw_position!r} is not long, short or flat"
            )

    order_id = payload.get("order_id")
    comment = payload.get("comment")
    return Signal(
        strategy=strategy.strip()[:64],
        ticker=ticker.strip(),
        action=_action(payload.get("action")),
        quantity=_decimal(payload.get("quantity"), "quantity"),
        target_position=_decimal(payload.get("target_position"), "target_position"),
        market_position=market_position,
        order_id=str(order_id).strip()[:64] if order_id else None,
        price=_decimal(payload.get("price"), "price"),
        bar_time=_timestamp(payload.get("bar_time"), "bar_time"),
        fired_at=_timestamp(payload.get("fired_at"), "fired_at"),
        comment=str(comment).strip()[:200] if comment else None,
    )
