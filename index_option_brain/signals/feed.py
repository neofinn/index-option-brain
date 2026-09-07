"""The pull side, for consumers that cannot be POSTed to.

An MT4/MT5 Expert Advisor has no listening socket. It cannot receive a
webhook; it calls `WebRequest` on a timer. So the relay's job for an EA is
not to deliver but to *hold* — and this router is what the EA reads.

Two formats, because MQL has no JSON parser
-------------------------------------------
`/signals` returns JSON for anything that can parse it. `/signals.csv`
returns a cursor line followed by one comma-separated row per signal,
which `StringSplit` handles in four lines of MQL. Shipping only JSON would
mean every EA author writing their own parser, and a hand-rolled JSON
parser in an order path is a bug with a position attached.

Only SENT rows are fed
----------------------
A route that is not enabled produces `DRY_RUN` rows, and those are
withheld here. That is the whole meaning of a dry run: the rehearsal is
recorded and auditable, and nothing downstream acts on it. Blocked and
failed rows are withheld for the same reason — an EA must never see a
signal the relay refused.
"""

from __future__ import annotations

import hmac
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from index_option_brain.database.models import SignalDispatchRow
from index_option_brain.integrations.webhooks.endpoints import Endpoint
from index_option_brain.signals.relay import Outcome, SignalRelay

#: The column order of the CSV feed. Fixed and documented, because an EA
#: reads it by index — inserting a column in the middle would silently
#: shift every field for every deployed EA.
CSV_COLUMNS = (
    "seq",
    "fired_at",
    "strategy",
    "symbol",
    "action",
    "quantity",
    "target",
    "price",
    "order_id",
)


def _read_token(request: Request) -> str | None:
    header = request.headers.get("authorization")
    if header and header.lower().startswith("bearer "):
        return header[7:].strip()
    token = request.query_params.get("token")
    return token.strip() if token else None


def _csv_cell(value: Any) -> str:
    """A CSV cell an EA can read positionally.

    Commas and newlines are stripped rather than quoted: MQL's
    `StringSplit` has no notion of quoting, so a quoted field would shift
    every column after it. The affected fields are free text a human
    wrote — a strategy name, a comment — and losing a comma there is
    better than misaligning an order row.
    """
    if value is None:
        return ""
    text = str(value)
    return text.replace(",", ";").replace("\n", " ").replace("\r", "")


def _row_cells(row: SignalDispatchRow) -> list[str]:
    return [
        _csv_cell(row.seq),
        _csv_cell(row.sent_at.isoformat() if row.sent_at else row.received_at.isoformat()),
        _csv_cell(row.strategy),
        _csv_cell(row.symbol),
        _csv_cell(row.action),
        _csv_cell("" if row.quantity is None else row.quantity),
        _csv_cell("" if row.target_position is None else row.target_position),
        _csv_cell(""),
        _csv_cell(""),
    ]


def _as_dict(row: SignalDispatchRow) -> dict[str, Any]:
    return {
        "seq": row.seq,
        "route": row.route,
        "strategy": row.strategy,
        "ticker": row.ticker,
        "symbol": row.symbol,
        "action": row.action,
        "quantity": None if row.quantity is None else str(row.quantity),
        "target_position": (
            None if row.target_position is None else str(row.target_position)
        ),
        "received_at": row.received_at.isoformat(),
        "sent_at": row.sent_at.isoformat() if row.sent_at else None,
    }


def create_signal_feed_router(
    relay: SignalRelay, endpoints: dict[str, Endpoint]
) -> APIRouter:
    """Read-only endpoints an EA or a script polls for relayed signals."""

    router = APIRouter()

    def authorize(request: Request, slug: str) -> Endpoint | JSONResponse:
        endpoint = endpoints.get(slug)
        if endpoint is None:
            return JSONResponse({"error": "no such endpoint"}, status_code=404)
        supplied = _read_token(request)
        if not supplied or not hmac.compare_digest(supplied, endpoint.read_token):
            return JSONResponse({"error": "read token required"}, status_code=401)
        return endpoint

    async def actionable(slug: str, cursor: int, limit: int) -> list[SignalDispatchRow]:
        rows = await relay.recent(slug, cursor=cursor, limit=limit)
        return [row for row in rows if row.outcome == Outcome.SENT]

    @router.get("/v1/{slug}/signals")
    async def signals(
        slug: str, request: Request, since: int = 0, limit: int = 50
    ) -> JSONResponse:
        endpoint = authorize(request, slug)
        if isinstance(endpoint, JSONResponse):
            return endpoint
        rows = await relay.recent(slug, cursor=since, limit=limit)
        actioned = [row for row in rows if row.outcome == Outcome.SENT]
        return JSONResponse(
            {
                "route": slug,
                "count": len(actioned),
                # Advanced past every row examined, not only the ones
                # returned. Advancing only past SENT rows would re-examine
                # a blocked signal on every poll for as long as it is the
                # newest thing in the table.
                "next_cursor": rows[-1].seq if rows else since,
                "signals": [_as_dict(row) for row in actioned],
            }
        )

    @router.get("/v1/{slug}/signals.csv", response_class=PlainTextResponse)
    async def signals_csv(
        slug: str, request: Request, since: int = 0, limit: int = 50
    ) -> PlainTextResponse:
        """Cursor line, header line, then one row per signal.

        The cursor comes first so an EA can read it without counting
        lines, and it is also returned in `X-Next-Cursor` for clients that
        would rather read a header.
        """
        endpoint = authorize(request, slug)
        if isinstance(endpoint, JSONResponse):
            return PlainTextResponse(
                "#error=unauthorized\n",
                status_code=endpoint.status_code,
            )
        rows = await relay.recent(slug, cursor=since, limit=limit)
        cursor = rows[-1].seq if rows else since
        lines = [f"#cursor={cursor}", ",".join(CSV_COLUMNS)]
        lines.extend(
            ",".join(_row_cells(row)) for row in rows if row.outcome == Outcome.SENT
        )
        return PlainTextResponse(
            "\n".join(lines) + "\n", headers={"X-Next-Cursor": str(cursor)}
        )

    @router.get("/v1/{slug}/routes")
    async def route_state(slug: str, request: Request) -> JSONResponse:
        """Whether this route would actually send, and what bounds it.

        `enabled` is reported because the difference between a rehearsal
        and a live order is invisible from the signal feed alone — a dry
        run simply produces no rows, which looks identical to a quiet
        strategy.
        """
        endpoint = authorize(request, slug)
        if isinstance(endpoint, JSONResponse):
            return endpoint
        route = relay.routes.get(slug)
        if route is None:
            return JSONResponse({"route": slug, "configured": False})
        return JSONResponse(
            {
                "route": slug,
                "configured": True,
                "enabled": route.enabled,
                "destination": route.destination.name,
                "symbols": sorted(route.symbol_map),
                "allowed_actions": sorted(str(a) for a in route.allowed_actions),
                "max_quantity": str(route.max_quantity),
                "max_orders_per_day": route.max_orders_per_day,
                "max_age_seconds": int(route.max_age.total_seconds()),
            }
        )

    return router
