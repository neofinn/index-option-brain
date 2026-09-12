"""What a TradingView alert is allowed to say, and what it becomes.

A TradingView webhook is an inbound HTTP POST fired by an alert on a chart.
This module fixes its meaning in one place: an alert is an **observation
about price on a chart**, it becomes an `Event`, and an `Event` means
"something changed; analyze it". The invariant that already governs the
internal trigger engine (spec §4) governs this one identically — nothing
here can reach a broker, a strike, or a size.

Three things a chart cannot say
-------------------------------
The `TriggerType` enum spans four families, and a chart can only observe
one of them. Pine Script sees open/high/low/close/volume for one symbol; it
does not see the fifty constituents of the index, and it does not see the
option chain. So an alert claiming `IV_EXPANSION_COLLAPSE` or
`BREADTH_CHANGE` is not a reading, it is an assertion about data the sender
could not have had — and the receiver refuses it rather than letting a
label smuggle unmeasured content into the pipeline. `CHART_OBSERVABLE`
below is that boundary, written down.

Symbols are mapped, never matched
---------------------------------
`"BANKNIFTY".startswith("NIFTY")` is false but `"BANKNIFTY" in ...` traps
are one careless line away, and a substring rule routes every BANKNIFTY
alert to NIFTY — a wrong index, silently, with a plausible price attached.
The map is exact and total: a ticker not in it is rejected, not guessed.

The secret never reaches this module
------------------------------------
TradingView cannot sign a request, so the shared secret travels inside the
alert body. `WebhookGuard` strips it before parsing, and `TradingViewAlert`
has no field that could hold it. Keeping the credential structurally
unrepresentable here is stronger than remembering not to log it: the event
that gets persisted, replayed and rendered on a console literally cannot
carry it.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict

from index_option_brain.contracts.enums import TriggerType
from index_option_brain.contracts.events import Event


class RejectionReason(StrEnum):
    """Why an alert was not accepted. Every path out of the receiver that is
    not an accepted event names one of these, so a silent drop is not
    representable."""

    BODY_TOO_LARGE = "BODY_TOO_LARGE"
    MALFORMED_BODY = "MALFORMED_BODY"
    MISSING_FIELD = "MISSING_FIELD"
    BAD_FIELD = "BAD_FIELD"
    BAD_SECRET = "BAD_SECRET"
    SOURCE_NOT_ALLOWED = "SOURCE_NOT_ALLOWED"
    UNKNOWN_SYMBOL = "UNKNOWN_SYMBOL"
    NOT_CHART_OBSERVABLE = "NOT_CHART_OBSERVABLE"
    STALE = "STALE"
    FUTURE_DATED = "FUTURE_DATED"
    DUPLICATE = "DUPLICATE"


class AlertRejected(Exception):
    """A rejection carries a reason code and a message safe to return.

    The message is deliberately built from field *names* and never from
    field *values*: echoing the body back would put the shared secret into
    an HTTP response and, from there, into TradingView's alert log.
    """

    def __init__(self, reason: RejectionReason, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else str(reason))


#: The trigger types a chart can actually observe. Everything outside this
#: set needs the constituent feed or the option chain, neither of which
#: Pine Script can see.
CHART_OBSERVABLE: frozenset[TriggerType] = frozenset(
    {
        TriggerType.SIGNIFICANT_PRICE_MOVEMENT,
        TriggerType.BREAKOUT,
        TriggerType.BREAKDOWN,
        TriggerType.VWAP_CROSSING,
        TriggerType.SUPPORT_RESISTANCE_TEST,
        TriggerType.OPENING_RANGE_EVENT,
        TriggerType.VOLATILITY_EXPANSION_CONTRACTION,
        TriggerType.VOLUME_ANOMALY,
    }
)

#: TradingView ticker (as `{{ticker}}` or `{{exchange}}:{{ticker}}` renders
#: it) to the symbol this system trades. Exact keys only.
TICKER_TO_SYMBOL: dict[str, str] = {
    "NIFTY": "NIFTY",
    "NIFTY1!": "NIFTY",
    "NSE:NIFTY": "NIFTY",
    "NSE:NIFTY1!": "NIFTY",
    "BANKNIFTY": "BANKNIFTY",
    "BANKNIFTY1!": "BANKNIFTY",
    "NSE:BANKNIFTY": "BANKNIFTY",
    "NSE:BANKNIFTY1!": "BANKNIFTY",
    "CNXBANK": "BANKNIFTY",
    "NSE:CNXBANK": "BANKNIFTY",
}

#: Stable namespace so a redelivered alert derives the same event id. Two
#: deliveries of one alert are one event, in every store that keys on id.
_NAMESPACE = uuid.UUID("6f4d0f3a-6a1e-5c39-9d5f-1f1b6d2a8c74")

_MAX_BODY_BYTES = 8192

#: How stale an alert may be. TradingView does not retry, so a late
#: delivery is a network fault or a replay, and acting on a ten-minute-old
#: breakout is worse than dropping it.
DEFAULT_MAX_AGE = timedelta(seconds=120)
DEFAULT_MAX_SKEW = timedelta(seconds=30)


def check_freshness(
    alert: TradingViewAlert,
    *,
    now: datetime,
    max_age: timedelta = DEFAULT_MAX_AGE,
    max_skew: timedelta = DEFAULT_MAX_SKEW,
) -> None:
    """Raise unless the alert fired recently enough to act on.

    Lives here rather than in `WebhookGuard` because there are two ways in
    — the standalone receiver and the gateway's `tradingview` endpoint —
    and the second one was written without this check. A rule with one
    definition and two callers cannot drift; two copies of it already had.
    """
    age = now - alert.fired_at
    if age > max_age:
        raise AlertRejected(
            RejectionReason.STALE,
            f"alert fired {int(age.total_seconds())}s ago",
        )
    if -age > max_skew:
        # A future-dated alert is either a clock the sender controls or a
        # fabricated body; both make the freshness window meaningless.
        raise AlertRejected(
            RejectionReason.FUTURE_DATED,
            "alert is dated in the future beyond the allowed skew",
        )


def resolve_symbol(ticker: str) -> str:
    """The engine symbol for a TradingView ticker, or raise.

    Case is normalized and surrounding whitespace stripped; nothing else is
    inferred. An unrecognised ticker is a configuration error the operator
    must see, not a symbol to guess at.
    """
    key = ticker.strip().upper()
    symbol = TICKER_TO_SYMBOL.get(key)
    if symbol is None:
        raise AlertRejected(
            RejectionReason.UNKNOWN_SYMBOL,
            "ticker is not in the configured TradingView symbol map",
        )
    return symbol


class TradingViewAlert(BaseModel):
    """One alert firing, after authentication and validation.

    Times are what TradingView substituted: `bar_time` from `{{time}}` (the
    bar the alert fired on) and `fired_at` from `{{timenow}}` (when it
    fired). Both matter and they are not the same — a "once per bar close"
    alert on a 5-minute chart has a `bar_time` up to five minutes behind
    `fired_at`, and dedupe keys on the bar while staleness keys on the fire.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    trigger_type: TriggerType
    ticker: str
    interval: str
    price: Decimal
    bar_time: datetime
    fired_at: datetime
    level: Decimal | None = None
    note: str | None = None

    @property
    def dedupe_key(self) -> str:
        """One alert firing, identified by what makes it distinct.

        The bar, not the fire time: TradingView can deliver the same bar's
        alert twice across a reconnect, with two different `{{timenow}}`
        values. Keying on the fire time would let that through as two
        events.
        """
        return f"{self.symbol}|{self.trigger_type}|{self.interval}|{self.bar_time.isoformat()}"

    def to_event(self) -> Event:
        """The `Event` this alert becomes.

        The payload is assembled field by field rather than dumped from the
        model, so a field added here can never reach the event by accident.
        `significance_score` is left None on purpose: significance is scored
        against the system's own market state by the significance filter,
        and a number supplied by the sender would be an unverified input
        deciding whether the pipeline wakes.
        """
        return Event(
            event_id=str(uuid.uuid5(_NAMESPACE, self.dedupe_key)),
            trigger_type=self.trigger_type,
            timestamp=self.fired_at,
            payload={
                "source": "tradingview",
                "symbol": self.symbol,
                "ticker": self.ticker,
                "interval": self.interval,
                "price": str(self.price),
                "bar_time": self.bar_time.isoformat(),
                "level": None if self.level is None else str(self.level),
                "note": self.note,
            },
            significance_score=None,
        )


def _require(payload: dict[str, Any], field: str) -> Any:
    if field not in payload or payload[field] is None:
        raise AlertRejected(RejectionReason.MISSING_FIELD, f"'{field}' is required")
    return payload[field]


def _decimal(value: Any, field: str) -> Decimal:
    """A price, from whatever JSON type TradingView produced.

    `{{close}}` substitutes a bare number, so the body is usually valid JSON
    with a float there; but an operator writing `"price": "{{close}}"` gets
    a string, and both should work. Floats route through `str()` rather than
    `Decimal(float)` so 24025.65 stays 24025.65 instead of acquiring
    seventeen digits of binary residue.
    """
    if isinstance(value, bool):
        raise AlertRejected(RejectionReason.BAD_FIELD, f"'{field}' is not a number")
    try:
        if isinstance(value, float | int):
            return Decimal(str(value))
        if isinstance(value, str):
            return Decimal(value.strip())
    except (InvalidOperation, ValueError) as exc:
        raise AlertRejected(RejectionReason.BAD_FIELD, f"'{field}' is not a number") from exc
    raise AlertRejected(RejectionReason.BAD_FIELD, f"'{field}' is not a number")


def _timestamp(value: Any, field: str) -> datetime:
    """A TradingView `{{time}}` / `{{timenow}}` value, always tz-aware UTC.

    TradingView emits ISO-8601 in UTC with a trailing `Z`. A naive datetime
    is treated as UTC rather than as local time: the receiver may run in any
    zone, and silently reinterpreting a UTC instant as IST would put every
    staleness check five and a half hours out.
    """
    if not isinstance(value, str):
        raise AlertRejected(RejectionReason.BAD_FIELD, f"'{field}' is not a timestamp")
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError as exc:
        raise AlertRejected(
            RejectionReason.BAD_FIELD, f"'{field}' is not an ISO-8601 timestamp"
        ) from exc
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _trigger_type(value: Any) -> TriggerType:
    if not isinstance(value, str):
        raise AlertRejected(RejectionReason.BAD_FIELD, "'kind' is not a trigger name")
    try:
        trigger = TriggerType(value.strip().upper())
    except ValueError as exc:
        raise AlertRejected(RejectionReason.BAD_FIELD, "'kind' is not a known trigger") from exc
    if trigger not in CHART_OBSERVABLE:
        raise AlertRejected(
            RejectionReason.NOT_CHART_OBSERVABLE,
            f"'{trigger}' needs data a chart cannot see (constituents or the option chain)",
        )
    return trigger


def alert_from_payload(payload: dict[str, Any]) -> TradingViewAlert:
    """Validate one authenticated alert body into a `TradingViewAlert`.

    Called only with the secret already removed by `WebhookGuard`.
    """
    ticker = _require(payload, "ticker")
    if not isinstance(ticker, str):
        raise AlertRejected(RejectionReason.BAD_FIELD, "'ticker' is not a string")
    interval = payload.get("interval") or "unknown"
    if not isinstance(interval, str):
        raise AlertRejected(RejectionReason.BAD_FIELD, "'interval' is not a string")
    note = payload.get("note")
    if note is not None and not isinstance(note, str):
        raise AlertRejected(RejectionReason.BAD_FIELD, "'note' is not a string")

    level_raw = payload.get("level")
    return TradingViewAlert(
        symbol=resolve_symbol(ticker),
        trigger_type=_trigger_type(_require(payload, "kind")),
        ticker=ticker.strip(),
        interval=interval.strip(),
        price=_decimal(_require(payload, "price"), "price"),
        bar_time=_timestamp(_require(payload, "bar_time"), "bar_time"),
        fired_at=_timestamp(_require(payload, "fired_at"), "fired_at"),
        level=None if level_raw is None else _decimal(level_raw, "level"),
        note=note.strip()[:280] if note else None,
    )
