"""The five ways a signal stops, and the one way it goes."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
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
    kill_switch_engaged,
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
            # Never {}: an empty environment resolves the kill-switch file
            # to var/RELAY_KILLED relative to the working directory, so the
            # day that file exists every one of these tests blocks and the
            # failure looks like a logic bug. Pointed at a path that cannot
            # exist instead.
            environ=environ
            if environ is not None
            else {"SIGNAL_RELAY_KILL_FILE": "/nonexistent/relay-killed"},
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


class TestExitAgainstAnHttpBroker:
    """The defect a live run against a mock broker exposed.

    Rendering the entry template for an exit produced
    `{"transactionType": "EXIT", "quantity": 0}` — not an order any broker
    accepts, and one a mock answers 200 to, so it looked like it worked
    right up until it mattered. Closing a position means buying or selling
    what is actually held, and the relay does not query the account.
    """

    async def test_an_exit_with_no_exit_template_is_refused(
        self, database: Database
    ) -> None:
        relay, http = relay_over(database)
        body = payload(action="exit")
        del body["quantity"]
        result = await relay.dispatch("tv-strategy", body)
        assert result.outcome == Outcome.BLOCKED
        assert "exit_body_template" in (result.reason or "")
        assert http.calls == []

    async def test_an_exit_template_is_used_when_present(
        self, database: Database
    ) -> None:
        destination = HttpDestination(
            url="https://api.broker.test/orders",
            headers={"access-token": BROKER_TOKEN},
            body_template={"side": "{action_upper}", "qty": "{quantity}"},
            exit_body_template={"squareOff": True, "securityId": "{symbol}"},
            exit_url="https://api.broker.test/positions/close",
        )
        relay, http = relay_over(
            database, routes={"tv-strategy": route(destination=destination)}
        )
        body = payload(action="exit")
        del body["quantity"]
        result = await relay.dispatch("tv-strategy", body)
        assert result.outcome == Outcome.SENT
        assert http.calls[0]["url"] == "https://api.broker.test/positions/close"
        assert http.calls[0]["json"] == {"squareOff": True, "securityId": "13"}

    async def test_an_entry_still_uses_the_entry_url(self, database: Database) -> None:
        destination = HttpDestination(
            url="https://api.broker.test/orders",
            body_template={"side": "{action_upper}"},
            exit_body_template={"squareOff": True},
            exit_url="https://api.broker.test/positions/close",
        )
        relay, http = relay_over(
            database, routes={"tv-strategy": route(destination=destination)}
        )
        await relay.dispatch("tv-strategy", payload())
        assert http.calls[0]["url"] == "https://api.broker.test/orders"

    async def test_an_exit_on_a_pull_route_is_fine(self, database: Database) -> None:
        """An EA reconciles to a target of zero, so "close everything" is
        fully expressible there — the restriction is HTTP-only."""
        relay, _ = relay_over(
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
        body = payload(action="exit")
        del body["quantity"]
        assert (await relay.dispatch("ea", body)).outcome == Outcome.SENT

    async def test_the_audited_url_is_the_url_transmitted(
        self, database: Database
    ) -> None:
        """The preview is what gets sent, not a second rendering — so what
        the audit row says was transmitted is what was."""
        destination = HttpDestination(
            url="https://api.broker.test/orders",
            body_template={"side": "{action_upper}"},
            exit_body_template={"squareOff": True},
            exit_url="https://api.broker.test/positions/close",
        )
        relay, http = relay_over(
            database, routes={"tv-strategy": route(destination=destination)}
        )
        body = payload(action="exit")
        del body["quantity"]
        result = await relay.dispatch("tv-strategy", body)
        assert result.request_preview is not None
        assert result.request_preview["url"] == http.calls[0]["url"]


class TestSenderVersusOperatorFault:
    """Which refusals the sender should see as failures.

    A rejection reported as a success puts a green tick in TradingView's
    alert log for an alert that did nothing. A policy refusal reported as
    a failure makes the operator's own guard look like a broken webhook.
    """

    async def test_a_malformed_payload_is_the_senders(
        self, database: Database
    ) -> None:
        relay, _ = relay_over(database)
        result = await relay.dispatch("tv-strategy", {"ticker": "NIFTY"})
        assert result.sender_error is True

    async def test_an_unmapped_ticker_is_the_senders(self, database: Database) -> None:
        relay, _ = relay_over(database)
        result = await relay.dispatch("tv-strategy", payload(ticker="BANKNIFTY"))
        assert result.sender_error is True

    async def test_a_quantity_ceiling_is_the_operators(
        self, database: Database
    ) -> None:
        relay, _ = relay_over(database)
        result = await relay.dispatch("tv-strategy", payload(quantity=99))
        assert result.sender_error is False

    async def test_a_daily_cap_is_the_operators(self, database: Database) -> None:
        relay, _ = relay_over(
            database, routes={"tv-strategy": route(max_orders_per_day=1)}
        )
        await relay.dispatch("tv-strategy", payload())
        result = await relay.dispatch(
            "tv-strategy", payload(bar_time="2026-09-07T04:11:00Z")
        )
        assert result.outcome == Outcome.BLOCKED
        assert result.sender_error is False

    async def test_the_kill_switch_is_the_operators(self, database: Database) -> None:
        relay, _ = relay_over(database, environ={KILL_SWITCH_ENV: "1"})
        result = await relay.dispatch("tv-strategy", payload())
        assert result.sender_error is False

    async def test_staleness_is_nobodys_fault(self, database: Database) -> None:
        """Usually the network. Reporting it as the sender's would have
        them hunting a template bug that is not there."""
        relay, _ = relay_over(database)
        result = await relay.dispatch(
            "tv-strategy", payload(fired_at="2026-09-07T04:00:00Z")
        )
        assert result.outcome == Outcome.BLOCKED
        assert result.sender_error is False


class TestTheKillSwitchFile:
    """The second way in, for when you are holding a phone.

    The environment variable cannot be set by anything but the relay's own
    process, so the chat bot engages the switch by creating a file. The
    asymmetry is deliberate: creating it is possible from outside, removing
    it is not.
    """

    def test_a_present_file_engages_it(self, tmp_path: Path) -> None:
        marker = tmp_path / "RELAY_KILLED"
        marker.write_text("")
        assert kill_switch_engaged({"SIGNAL_RELAY_KILL_FILE": str(marker)}) is True

    def test_an_absent_file_does_not(self, tmp_path: Path) -> None:
        assert (
            kill_switch_engaged(
                {"SIGNAL_RELAY_KILL_FILE": str(tmp_path / "nothing")}
            )
            is False
        )

    def test_the_env_var_still_works_on_its_own(self, tmp_path: Path) -> None:
        assert kill_switch_engaged(
            {
                KILL_SWITCH_ENV: "1",
                "SIGNAL_RELAY_KILL_FILE": str(tmp_path / "nothing"),
            }
        ) is True

    async def test_a_live_route_stops_when_the_file_appears(
        self, database: Database, tmp_path: Path
    ) -> None:
        marker = tmp_path / "RELAY_KILLED"
        relay, http = relay_over(
            database, environ={"SIGNAL_RELAY_KILL_FILE": str(marker)}
        )
        assert (await relay.dispatch("tv-strategy", payload())).outcome == Outcome.SENT
        marker.write_text("stopped from chat")
        result = await relay.dispatch(
            "tv-strategy", payload(bar_time="2026-09-07T04:11:00Z")
        )
        assert result.outcome == Outcome.BLOCKED
        assert len(http.calls) == 1

    def test_an_unreadable_path_engages_it(self) -> None:
        """An unreadable path is not a reason to start trading."""
        assert kill_switch_engaged({"SIGNAL_RELAY_KILL_FILE": "\x00bad"}) is True
