"""The five ways a signal stops, and the one way it goes."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from index_option_brain.data.http import HttpError, HttpResponse
from index_option_brain.database.engine import Database
from index_option_brain.database.models import SignalDispatchRow
from index_option_brain.signals.contract import SignalAction, signal_from_payload
from index_option_brain.signals.relay import (
    KILL_SWITCH_ENV,
    Outcome,
    SignalRelay,
    render_body,
    template_fields,
)
from index_option_brain.signals.routes import (
    HttpDestination,
    PullDestination,
    SignalRoute,
)

NOW = datetime(2026, 9, 7, 4, 15, 30, tzinfo=UTC)


class FakeSession:
    """Records what the relay tried to transmit."""

    def __init__(self, status: int = 200, body: str = '{"orderId":"X1"}') -> None:
        self.status = status
        self.body = body
        self.calls: list[dict[str, Any]] = []
        self.raise_with: Exception | None = None

    async def post(
        self,
        url: str,
        *,
        json: Any | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        if self.raise_with is not None:
            raise self.raise_with
        self.calls.append({"url": url, "json": json, "headers": dict(headers or {})})
        return HttpResponse(status_code=self.status, text=self.body)

    async def get(self, url: str, **kw: Any) -> HttpResponse:  # pragma: no cover
        raise AssertionError("the relay must never GET")

    async def delete(self, url: str, **kw: Any) -> HttpResponse:  # pragma: no cover
        raise AssertionError("the relay must never DELETE")

    async def aclose(self) -> None:
        return None


def payload(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "strategy": "orb-nifty",
        "ticker": "NIFTY",
        "action": "buy",
        "quantity": 1,
        "bar_time": "2026-09-07T04:10:00Z",
        "fired_at": "2026-09-07T04:15:00Z",
    }
    body.update(overrides)
    return body


#: Distinctive on purpose. An earlier version of the audit-row test used
#: "tok", which is a substring of the header *name* "access-token" — so it
#: passed on a bug and failed on correct code.
BROKER_TOKEN = "zzsecretvaluenotinauditzz"

BROKER = HttpDestination(
    url="https://api.broker.test/orders",
    headers={"access-token": BROKER_TOKEN},
    body_template={
        "transactionType": "{action_upper}",
        "securityId": "{symbol}",
        "quantity": "{quantity}",
        "orderType": "MARKET",
    },
)


def route(**overrides: object) -> SignalRoute:
    kwargs: dict[str, Any] = {
        "name": "tv-strategy",
        "destination": BROKER,
        "symbol_map": {"NIFTY": "13"},
        "enabled": True,
        "max_quantity": Decimal(2),
    }
    kwargs.update(overrides)
    return SignalRoute(**kwargs)


@pytest.fixture
async def database() -> AsyncIterator[Database]:
    db = Database.in_memory()
    await db.create_schema()
    yield db
    await db.aclose()


def relay_over(
    database: Database,
    *,
    routes: dict[str, SignalRoute] | None = None,
    session: FakeSession | None = None,
    environ: Mapping[str, str] | None = None,
    clock: Any = None,
) -> tuple[SignalRelay, FakeSession]:
    http = session or FakeSession()
    return (
        SignalRelay(
            routes or {"tv-strategy": route()},
            database,
            session=http,
            clock=clock or (lambda: NOW),
            environ=environ if environ is not None else {},
        ),
        http,
    )


async def rows(database: Database) -> list[SignalDispatchRow]:
    async with database.session() as session:
        return list(
            (await session.execute(SignalDispatchRow.__table__.select())).all()  # type: ignore[arg-type]
        )


class TestItSends:
    async def test_an_enabled_route_transmits_the_rendered_body(
        self, database: Database
    ) -> None:
        relay, http = relay_over(database)
        result = await relay.dispatch("tv-strategy", payload())
        assert result.outcome == Outcome.SENT
        assert http.calls[0]["url"] == "https://api.broker.test/orders"
        assert http.calls[0]["json"] == {
            "transactionType": "BUY",
            "securityId": "13",
            # A bare numeric placeholder becomes a JSON number: several
            # broker APIs reject "1" where they want 1.
            "quantity": 1,
            "orderType": "MARKET",
        }
        assert http.calls[0]["headers"] == {"access-token": BROKER_TOKEN}

    async def test_the_audit_row_keeps_the_response(self, database: Database) -> None:
        relay, _ = relay_over(database)
        await relay.dispatch("tv-strategy", payload())
        stored = await rows(database)
        assert stored[0]._mapping["outcome"] == Outcome.SENT
        assert stored[0]._mapping["response_status"] == 200

    async def test_no_credential_reaches_the_audit_row(self, database: Database) -> None:
        """Header names, never header values. Knowing an access-token
        header was set is what an operator needs when a broker answers
        401; the value is the one thing that must not be stored."""
        relay, _ = relay_over(database)
        await relay.dispatch("tv-strategy", payload())
        stored = await rows(database)
        assert BROKER_TOKEN not in str(stored[0]._mapping["request_preview"])
        assert "access-token" in str(stored[0]._mapping["request_preview"])


class TestTheFiveWaysItStops:
    async def test_the_kill_switch_wins_before_anything_is_parsed(
        self, database: Database
    ) -> None:
        relay, http = relay_over(database, environ={KILL_SWITCH_ENV: "1"})
        result = await relay.dispatch("tv-strategy", {"nonsense": True})
        assert result.outcome == Outcome.BLOCKED
        assert KILL_SWITCH_ENV in (result.reason or "")
        assert http.calls == []

    async def test_a_disabled_route_rehearses_in_full(self, database: Database) -> None:
        """A dry run that exercises nothing is a mode that first runs on
        the day it matters. This one renders exactly what it would have
        sent and records it."""
        relay, http = relay_over(
            database, routes={"tv-strategy": route(enabled=False)}
        )
        result = await relay.dispatch("tv-strategy", payload())
        assert result.outcome == Outcome.DRY_RUN
        assert http.calls == []
        assert result.request_preview is not None
        assert result.request_preview["body"]["securityId"] == "13"

    async def test_a_quantity_over_the_ceiling_is_blocked(
        self, database: Database
    ) -> None:
        relay, http = relay_over(database)
        result = await relay.dispatch("tv-strategy", payload(quantity=5))
        assert result.outcome == Outcome.BLOCKED
        assert "ceiling" in (result.reason or "")
        assert http.calls == []

    async def test_a_target_over_the_ceiling_is_blocked_too(
        self, database: Database
    ) -> None:
        """Checked on the absolute value: a target of -5 is as large a
        position as +5."""
        relay, _ = relay_over(database)
        body = payload(target_position=-5)
        del body["quantity"]
        result = await relay.dispatch("tv-strategy", body)
        assert result.outcome == Outcome.BLOCKED

    async def test_an_unmapped_ticker_is_blocked(self, database: Database) -> None:
        """A chart switched to the wrong symbol would otherwise send a real
        order for an instrument nobody meant."""
        relay, _ = relay_over(database)
        result = await relay.dispatch("tv-strategy", payload(ticker="BANKNIFTY"))
        assert result.outcome == Outcome.BLOCKED
        assert "symbol map" in (result.reason or "")

    async def test_a_disallowed_action_is_blocked(self, database: Database) -> None:
        relay, _ = relay_over(
            database,
            routes={
                "tv-strategy": route(
                    allowed_actions=frozenset({SignalAction.EXIT})
                )
            },
        )
        assert (await relay.dispatch("tv-strategy", payload())).outcome == Outcome.BLOCKED

    async def test_a_stale_signal_is_blocked(self, database: Database) -> None:
        """Acting on a five-minute-old entry is how a relay buys the top of
        a move that already finished."""
        relay, _ = relay_over(database)
        result = await relay.dispatch(
            "tv-strategy", payload(fired_at="2026-09-07T04:05:00Z")
        )
        assert result.outcome == Outcome.BLOCKED
        assert "ago" in (result.reason or "")

    async def test_a_future_dated_signal_is_blocked(self, database: Database) -> None:
        result = (
            await relay_over(database)[0].dispatch(
                "tv-strategy", payload(fired_at="2026-09-07T05:15:00Z")
            )
        )
        assert result.outcome == Outcome.BLOCKED

    async def test_a_duplicate_is_not_sent_twice(self, database: Database) -> None:
        """The database's unique constraint, not a memory set — so a
        restart between two deliveries of one alert cannot double-order."""
        relay, http = relay_over(database)
        first = await relay.dispatch("tv-strategy", payload())
        second = await relay.dispatch(
            "tv-strategy", payload(fired_at="2026-09-07T04:15:20Z")
        )
        assert first.outcome == Outcome.SENT
        assert second.outcome == Outcome.DUPLICATE
        assert len(http.calls) == 1

    async def test_a_duplicate_counts_as_handled_not_failed(
        self, database: Database
    ) -> None:
        relay, _ = relay_over(database)
        await relay.dispatch("tv-strategy", payload())
        assert (await relay.dispatch("tv-strategy", payload())).ok

    async def test_the_daily_cap_stops_a_runaway_strategy(
        self, database: Database
    ) -> None:
        relay, http = relay_over(
            database, routes={"tv-strategy": route(max_orders_per_day=2)}
        )
        for bar in range(4):
            await relay.dispatch(
                "tv-strategy", payload(bar_time=f"2026-09-07T04:1{bar}:00Z")
            )
        assert len(http.calls) == 2

    async def test_blocked_signals_do_not_consume_the_daily_cap(
        self, database: Database
    ) -> None:
        """The cap bounds orders, not attempts. A misconfigured template
        must not exhaust the allowance without one order reaching anyone."""
        relay, http = relay_over(
            database, routes={"tv-strategy": route(max_orders_per_day=1)}
        )
        await relay.dispatch("tv-strategy", payload(quantity=99))
        await relay.dispatch("tv-strategy", payload(bar_time="2026-09-07T04:11:00Z"))
        assert len(http.calls) == 1


class TestFailure:
    async def test_a_non_2xx_response_is_recorded_and_not_retried(
        self, database: Database
    ) -> None:
        """A broker call whose response was lost may well have been
        executed. A blind retry is how one intent becomes two positions."""
        relay, http = relay_over(
            database, session=FakeSession(status=400, body='{"error":"bad qty"}')
        )
        result = await relay.dispatch("tv-strategy", payload())
        assert result.outcome == Outcome.FAILED
        assert result.response_status == 400
        assert len(http.calls) == 1
        stored = await rows(database)
        assert "bad qty" in str(stored[0]._mapping["response_body"])

    async def test_a_transport_error_is_recorded(self, database: Database) -> None:
        session = FakeSession()
        session.raise_with = HttpError("connection reset")
        relay, _ = relay_over(database, session=session)
        result = await relay.dispatch("tv-strategy", payload())
        assert result.outcome == Outcome.FAILED
        assert "connection reset" in (result.reason or "")

    async def test_a_failure_does_not_free_the_idempotency_key(
        self, database: Database
    ) -> None:
        """Deliberate. A retry of a failed send is an operator decision,
        because the previous attempt may have reached the broker."""
        relay, _ = relay_over(database, session=FakeSession(status=500))
        await relay.dispatch("tv-strategy", payload())
        assert (await relay.dispatch("tv-strategy", payload())).outcome == Outcome.DUPLICATE

    async def test_a_relay_with_no_session_fails_rather_than_pretending(
        self, database: Database
    ) -> None:
        relay = SignalRelay(
            {"tv-strategy": route()}, database, clock=lambda: NOW, environ={}
        )
        assert (await relay.dispatch("tv-strategy", payload())).outcome == Outcome.FAILED

    async def test_an_unknown_route_is_blocked(self, database: Database) -> None:
        relay, _ = relay_over(database)
        assert (await relay.dispatch("nope", payload())).outcome == Outcome.BLOCKED


class TestPullDestination:
    async def test_it_makes_no_outbound_call(self, database: Database) -> None:
        """An EA has no listening socket. "Sent" means the signal is
        durable and on the pull endpoint."""
        relay, http = relay_over(
            database,
            routes={
                "ea": SignalRoute(
                    name="ea",
                    destination=PullDestination(),
                    symbol_map={"NIFTY": "NIFTY.I"},
                    enabled=True,
                )
            },
        )
        result = await relay.dispatch("ea", payload())
        assert result.outcome == Outcome.SENT
        assert http.calls == []

    async def test_a_disabled_pull_route_produces_no_actionable_signal(
        self, database: Database
    ) -> None:
        """Which is the point of a dry run: the rehearsal is recorded and
        the EA sees nothing."""
        relay, _ = relay_over(
            database,
            routes={
                "ea": SignalRoute(
                    name="ea",
                    destination=PullDestination(),
                    symbol_map={"NIFTY": "NIFTY.I"},
                )
            },
        )
        await relay.dispatch("ea", payload())
        feed = await relay.recent("ea")
        assert [row.outcome for row in feed] == [Outcome.DRY_RUN]


class TestAuditTrail:
    async def test_refusals_are_written_down(self, database: Database) -> None:
        """An audit trail recording only what was sent cannot answer the
        question actually asked after a bad day."""
        relay, _ = relay_over(database)
        await relay.dispatch("tv-strategy", payload(quantity=99))
        stored = await rows(database)
        assert stored[0]._mapping["outcome"] == Outcome.BLOCKED
        assert "ceiling" in str(stored[0]._mapping["reason"])

    async def test_a_malformed_payload_is_written_down_too(
        self, database: Database
    ) -> None:
        relay, _ = relay_over(database)
        await relay.dispatch("tv-strategy", {"ticker": "NIFTY"})
        stored = await rows(database)
        assert stored[0]._mapping["outcome"] == Outcome.BLOCKED

    async def test_the_cursor_feed_never_skips_or_repeats(
        self, database: Database
    ) -> None:
        relay, _ = relay_over(
            database, routes={"tv-strategy": route(max_orders_per_day=99)}
        )
        for bar in range(3):
            await relay.dispatch(
                "tv-strategy", payload(bar_time=f"2026-09-07T04:1{bar}:00Z")
            )
        first = await relay.recent("tv-strategy", limit=2)
        second = await relay.recent("tv-strategy", cursor=first[-1].seq)
        assert [r.seq for r in first] + [r.seq for r in second] == sorted(
            r.seq for r in first + second
        )
        assert len(second) == 1


class TestTemplates:
    def test_a_bare_numeric_placeholder_becomes_a_number(self) -> None:
        signal = signal_from_payload(payload(quantity=2))
        body = render_body(
            {"qty": "{quantity}"}, template_fields(signal, "13")
        )
        assert body == {"qty": 2}

    def test_an_integral_decimal_does_not_serialise_as_a_float(self) -> None:
        """Several broker APIs reject 2.0 where they want 2."""
        signal = signal_from_payload(payload(quantity=2))
        assert render_body({"q": "{quantity}"}, template_fields(signal, "13"))["q"] == 2

    def test_a_placeholder_inside_text_stays_text(self) -> None:
        signal = signal_from_payload(payload(quantity=2))
        body = render_body({"tag": "lots={quantity}"}, template_fields(signal, "13"))
        assert body["tag"] == "lots=2"

    def test_nested_templates_are_rendered(self) -> None:
        signal = signal_from_payload(payload())
        body = render_body(
            {"order": {"side": "{action_upper}", "legs": ["{symbol}"]}},
            template_fields(signal, "13"),
        )
        assert body == {"order": {"side": "BUY", "legs": ["13"]}}

    def test_an_unknown_placeholder_is_refused(self) -> None:
        """A template language with attribute access would make this file
        compute things; the reason it is configuration is that it does
        not."""
        signal = signal_from_payload(payload())
        with pytest.raises(Exception, match="does not"):
            render_body({"x": "{account_balance}"}, template_fields(signal, "13"))

    def test_both_side_spellings_are_offered(self) -> None:
        """So a template needs no conditional to produce what a given
        broker calls a side."""
        fields = template_fields(signal_from_payload(payload()), "13")
        assert fields["side_bs"] == "B"
        assert fields["side_long_short"] == "LONG"


class TestRouteConfig:
    def test_plaintext_http_to_a_remote_host_is_refused(self) -> None:
        """The headers carry a broker token, and a relay is exactly the
        component nobody looks at again after it starts working."""
        with pytest.raises(ValueError, match="plaintext"):
            HttpDestination(url="http://api.broker.test/orders", body_template={"a": 1})

    def test_loopback_http_is_allowed_for_a_local_bridge(self) -> None:
        assert HttpDestination(
            url="http://127.0.0.1:9000/order", body_template={"a": 1}
        ).url

    def test_a_route_without_a_symbol_map_is_refused(self) -> None:
        with pytest.raises(ValueError, match="symbol_map"):
            SignalRoute(name="x", destination=PullDestination())

    def test_a_route_is_disabled_by_default(self) -> None:
        """The single most important default in this package."""
        assert (
            SignalRoute(
                name="x", destination=PullDestination(), symbol_map={"NIFTY": "N"}
            ).enabled
            is False
        )

    def test_the_default_age_ceiling_is_tight(self) -> None:
        assert SignalRoute(
            name="x", destination=PullDestination(), symbol_map={"NIFTY": "N"}
        ).max_age <= timedelta(minutes=2)
