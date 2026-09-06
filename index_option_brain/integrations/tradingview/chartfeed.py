"""One pasteable line carrying everything a chart cannot compute.

The Pine mirror reproduces the Index brain exactly, because that brain is
OHLC arithmetic. The other three are not: breadth needs fifty quotes and
their weights, and open interest, implied volatility, the parity forward
and the volatility risk premium all need the option chain. Pine cannot
fetch any of it.

So it is *transported* instead of computed. `GET /api/chartfeed/{symbol}`
returns a single line the operator pastes into the indicator's settings,
and the chart renders those values beside the ones it worked out itself.
The chart never claims to have derived them — the panel labels the block
with the feed's age, and a feed older than the staleness window is drawn
in red and its values are withheld.

Why a string and not JSON
-------------------------
It has to survive a copy-paste into a TradingView input box: one line, no
newlines, no quotes to be mangled, short enough to read at a glance. Pipe
separated `key=value`, with a version tag first so a later format cannot be
misread by an older indicator.

Absence is absence
------------------
A field the engine could not measure is **omitted**, never sent as zero.
A max pain of 0 would render as a level at zero on the chart; an omitted
one renders as "—". This is the same rule the rest of the system runs on,
and it matters more here than usual because the consumer is a chart, where
a number is a line and a missing number is nothing at all.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import datetime
from decimal import Decimal
from typing import Any

#: Bumped whenever a key changes meaning. The indicator refuses a version
#: it does not know rather than reading the wrong field.
FEED_VERSION = "v1"

_SEPARATOR = "|"


def _number(value: Any, places: int = 2) -> str | None:
    """Format a measured number, or None when there is nothing to say."""
    if value is None:
        return None
    if isinstance(value, Decimal):
        value = float(value)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    if math.isnan(value):
        # A NaN reaching a chart draws nothing but poisons every
        # comparison it touches on the way there.
        return None
    return f"{value:.{places}f}"


def _levels(values: Iterable[Any], limit: int = 3) -> str | None:
    """A comma-separated level list, or None when there are none.

    Commas rather than a repeated key so the whole field survives one
    `str.split` in Pine, which has no regular expressions.
    """
    formatted = [n for n in (_number(v) for v in list(values)[:limit]) if n]
    return ",".join(formatted) if formatted else None


def encode(fields: dict[str, str | None], *, as_of: datetime, symbol: str) -> str:
    """Assemble the line. Keys with a None value are dropped entirely."""
    head = [
        FEED_VERSION,
        as_of.isoformat(timespec="seconds").replace("+00:00", "Z"),
        symbol.upper(),
    ]
    body = [f"{key}={value}" for key, value in fields.items() if value is not None]
    return _SEPARATOR.join([*head, *body])


def decode(line: str) -> dict[str, str]:
    """Read a feed line back into its fields. For tests and for the console.

    Returns an empty mapping for anything that is not a feed line of a
    version this build knows — a malformed paste must not half-parse into
    plausible numbers.
    """
    parts = [part for part in line.strip().split(_SEPARATOR) if part]
    if len(parts) < 3 or parts[0] != FEED_VERSION:
        return {}
    fields = {"version": parts[0], "as_of": parts[1], "symbol": parts[2]}
    for part in parts[3:]:
        key, sep, value = part.partition("=")
        if sep and key:
            fields[key] = value
    return fields


def build(result: Any) -> str:
    """The feed line for one completed analysis cycle.

    Reads a `BrainCycleResult` structurally rather than by import, so this
    module stays outside the pipeline's dependency graph — the same reason
    the receiver holds no engine.

    `auth` is emitted on every line, including when it is 0. Authorization
    is the one field whose absence would be read as "probably fine": a
    chart showing entry levels with no authorization line looks exactly
    like a chart showing an authorized trade.
    """
    state = result.state
    analysis = result.analysis
    index = analysis.index
    options = analysis.options
    volatility = analysis.volatility
    constituents = analysis.constituents

    fields: dict[str, str | None] = {
        "spot": _number(state.index_state.quote.ltp),
        "regime": str(result.regime.regime) if result.regime else None,
        "rgc": _number(result.regime.confidence if result.regime else None),
        # Upper-cased because the indicator compares it against literal
        # "BULLISH"/"BEARISH" to colour the row, and Direction's values are
        # lower case. A silent case mismatch would leave every signal grey.
        "dir": str(result.signal.direction).upper(),
        "sig": _number(result.signal.score),
        # Index brain — sent even though the chart recomputes it, because a
        # disagreement between the two is the cheapest possible check that
        # the transliteration is still faithful.
        "idx": _number(index.confidence),
        "sup": _levels(index.support_levels),
        "res": _levels(index.resistance_levels),
        # Constituents: fifty quotes and their weights, invisible to Pine.
        "adv": str(constituents.advances) if constituents.advances is not None else None,
        "dec": str(constituents.declines) if constituents.declines is not None else None,
        "brd": _number(constituents.breadth_score),
        "cov": _number(constituents.weight_coverage),
        # The option chain, all of it invisible to Pine.
        "mp": _number(options.max_pain_strike),
        "cw": _levels(options.call_walls),
        "pw": _levels(options.put_walls),
        "pcr": _number(options.pcr_oi),
        "xb": _number(options.excess_basis),
        "iv": _number(volatility.atm_iv),
        "rv": _number(volatility.realized_volatility),
        "rvw": (
            str(volatility.realized_window)
            if volatility.realized_window is not None
            else None
        ),
        "vrp": _number(volatility.volatility_risk_premium),
        "ivp": _number(volatility.iv_percentile),
        "em": _number(volatility.expected_move),
        "eam": _number(volatility.expected_absolute_move),
        "strat": str(result.selected_strategy),
        "auth": "1" if result.is_authorized else "0",
    }
    return encode(fields, as_of=state.timestamp, symbol=state.index_symbol)
