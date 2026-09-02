"""The Dhan adapter.

An honest note on what these tests are worth. The payloads below are built
from Dhan's published documentation, **not recorded from the live API** — that
needs a token this repository does not have. So they prove the adapter reads
the shape it was written for; they cannot prove that shape is what Dhan sends.

That distinction is the reason `scripts/dhan_probe.py` exists, and the reason
`DHAN_DESCRIPTOR.verified` is False. The NSE adapter is the cautionary tale:
its documented `bidprice` field is always null and the real top of book lives
in `buyPrice1`, so an adapter tested only against documented shapes would
have passed every test and produced a chain with no bid anywhere.

What these tests *do* protect for real: the auth headers, both error
envelopes (verified against the live hosts), the sandbox route gap (also
verified), the unit conversions, and every place the adapter refuses to guess.
"""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from index_option_brain.contracts.enums import (
    BarInterval,
    OptionType,
    OrderLifecycleState,
    OrderSide,
)
from index_option_brain.contracts.order import OrderRequest
from index_option_brain.data.adapters.base import DataAdapterError
from index_option_brain.data.adapters.dhan import (
    LIVE_BASE,
    SANDBOX_BASE,
    DhanApiError,
    DhanBrokerAdapter,
    DhanClient,
    DhanConfig,
    DhanMarketDataAdapter,
)
from index_option_brain.data.dhan_instruments import (
    DhanInstrumentMaster,
    parse_scrip_master,
)
from index_option_brain.data.http import HttpResponse, RecordedSession

RECORDED_MASTER = Path(__file__).parent / "recorded" / "dhan_scrip_master.csv"
NEAR_EXPIRY = date(2026, 9, 8)


@pytest.fixture
def master() -> DhanInstrumentMaster:
    return DhanInstrumentMaster(records=parse_scrip_master(RECORDED_MASTER.read_text()))


def config(*, sandbox: bool = False, **kwargs) -> DhanConfig:
    return DhanConfig(
        client_id="C123", access_token="tok", sandbox=sandbox, **kwargs
    )


def client(routes: dict, *, sandbox: bool = False, **kwargs) -> DhanClient:
    return DhanClient(config(sandbox=sandbox, **kwargs), RecordedSession(routes))


# --- Payloads shaped from Dhan's documentation, not recorded. ---------------

FUNDS = json.dumps(
    {
        "dhanClientId": "C123",
        "availabelBalance": 1487500.25,
        "sodLimit": 2000000.0,
        "utilizedAmount": 512499.75,
    }
)

CANDLES = json.dumps(
    {
        "open": [23800.0, 23850.0, 23900.0],
        "high": [23880.0, 23930.0, 23960.0],
        "low": [23760.0, 23820.0, 23870.0],
        "close": [23850.0, 23900.0, 23914.45],
        "volume": [0, 0, 0],
        # 27, 28 and 31 Aug 2026, 09:15 IST.
        "timestamp": [1787887800, 1787974200, 1788233400],
    }
)

CHAIN = json.dumps(
    {
        "status": "success",
        "data": {
            "last_price": 23914.45,
            "oc": {
                "23900.000000": {
                    "ce": {
                        "last_price": 131.6,
                        "top_bid_price": 130.15,
                        "top_ask_price": 131.6,
                        "volume": 2635119,
                        "oi": 95996,
                        "previous_oi": 21027,
                        "implied_volatility": 8.39,
                        "greeks": {
                            "delta": 0.5601,
                            "gamma": 0.0014,
                            "theta": -11.4,
                            "vega": 12.08,
                        },
                    },
                    "pe": {
                        "last_price": 114.4,
                        "top_bid_price": 114.4,
                        "top_ask_price": 115.0,
                        "volume": 2958819,
                        "oi": 90549,
                        "previous_oi": 51524,
                        "implied_volatility": 11.43,
                    },
                },
                "24000.000000": {
                    "ce": {
                        "last_price": 82.0,
                        "top_bid_price": 81.5,
                        "top_ask_price": 82.5,
                        "volume": 1_200_000,
                        "oi": 60000,
                        "previous_oi": 75000,
                        "implied_volatility": 8.9,
                    }
                },
            },
        },
    }
)


class TestConfiguration:
    def test_it_defaults_to_the_sandbox(self):
        """A mis-set flag on a trading adapter is not a configuration error,
        it is a trade."""
        assert DhanConfig(client_id="C", access_token="t").sandbox is True
        assert DhanConfig(client_id="C", access_token="t").base_url == SANDBOX_BASE

    def test_pointing_at_live_is_explicit(self):
        assert config(sandbox=False).base_url == LIVE_BASE
        assert config(sandbox=False).is_live

    def test_the_auth_headers_carry_both_credentials(self):
        headers = client({}).headers
        assert headers["access-token"] == "tok"
        assert headers["client-id"] == "C123"

    def test_broker_greeks_are_not_trusted_by_default(self):
        """Greeks depend on the rate and day-count used to compute them, and
        Dhan publishes neither. Mixing its numbers with locally computed ones
        would put two conventions in one delta-fit ranking."""
        assert config().trust_broker_greeks is False


class TestErrorEnvelopes:
    """Both shapes were observed on the live hosts. A client understanding
    only one would read the other as a successful response."""

    async def test_the_charts_family_envelope_is_recognized(self):
        body = json.dumps(
            {
                "errorType": "Invalid_Authentication",
                "errorCode": "DH-901",
                "errorMessage": "Client ID or user generated access token is invalid or expired.",
            }
        )
        c = client({"/fundlimit": HttpResponse(401, body)})
        with pytest.raises(DhanApiError) as exc:
            await c.get("/fundlimit")
        assert exc.value.code == "DH-901"
        assert "invalid or expired" in str(exc.value)

    async def test_the_chain_family_envelope_is_recognized(self):
        body = json.dumps({"Data": {"810": "ClientId is invalid"}, "status": "failed"})
        c = client({"/optionchain": HttpResponse(401, body)})
        with pytest.raises(DhanApiError) as exc:
            await c.post("/optionchain", {})
        assert exc.value.code == "810"

    async def test_a_failure_reported_inside_a_200_is_still_a_failure(self):
        """Dhan reports some failures in a 200 body. Checking the status code
        first would parse an error envelope as market data."""
        body = json.dumps({"Data": {"810": "ClientId is invalid"}, "status": "failed"})
        c = client({"/optionchain": HttpResponse(200, body)})
        with pytest.raises(DhanApiError):
            await c.post("/optionchain", {})

    async def test_the_code_is_preserved_because_codes_are_actionable(self):
        """DH-901 needs re-authentication, DH-905 needs a code fix. Collapsing
        them into one message loses the difference."""
        body = json.dumps(
            {"errorType": "Input_Exception", "errorCode": "DH-905", "errorMessage": "bad"}
        )
        c = client({"/charts": HttpResponse(400, body)})
        with pytest.raises(DhanApiError) as exc:
            await c.post("/charts/historical", {})
        assert exc.value.code == "DH-905"

    async def test_a_non_json_body_raises_rather_than_being_parsed(self):
        c = client({"/fundlimit": HttpResponse(502, "<html>bad gateway</html>")})
        with pytest.raises(DataAdapterError, match="non-JSON"):
            await c.get("/fundlimit")

    async def test_a_plain_404_is_reported_with_its_status(self):
        """The sandbox's Spring-default 404 carries no Dhan envelope, so it
        must fall through to the status check rather than being mistaken for
        one."""
        body = json.dumps({"timestamp": 1788383784351, "status": 404, "error": "Not Found"})
        c = client({"/optionchain": HttpResponse(404, body)})
        with pytest.raises(DataAdapterError, match="HTTP 404"):
            await c.post("/optionchain", {})


class TestSandboxRouteGap:
    """Probed on both hosts: the sandbox serves orders, funds and charts but
    404s on market feed and option chain."""

    async def test_market_data_on_the_sandbox_fails_with_an_explanation(self):
        c = client({}, sandbox=True)
        with pytest.raises(DataAdapterError) as exc:
            await c.post("/optionchain", {})
        assert "sandbox does not implement" in str(exc.value)
        assert "live host" in str(exc.value)

    async def test_the_marketfeed_family_is_covered_too(self):
        c = client({}, sandbox=True)
        with pytest.raises(DataAdapterError, match="does not implement"):
            await c.post("/marketfeed/ohlc", {})

    async def test_charts_and_orders_are_allowed_on_the_sandbox(self):
        """Which is what makes the order path testable with no money at
        risk."""
        c = client({"/charts/historical": CANDLES, "/orders": "{}"}, sandbox=True)
        assert await c.post("/charts/historical", {})
        assert await c.post("/orders", {}) == {}

    async def test_the_live_host_allows_market_data(self):
        c = client({"/optionchain": CHAIN}, sandbox=False)
        assert await c.post("/optionchain", {})


class TestAccount:
    async def test_funds_map_to_an_account_snapshot(self, master):
        """The thing that unblocks the Risk Engine."""
        adapter = DhanMarketDataAdapter(client({"/fundlimit": FUNDS}), master)
        account = await adapter.get_account_snapshot()
        assert account.available_margin == Decimal("1487500.25")
        assert account.used_margin == Decimal("512499.75")
        assert account.net_equity == Decimal("2000000.0")

    async def test_a_missing_field_raises_naming_it(self, master):
        """No defaults. A substituted zero reaches a sizing calculation
        looking like a measurement."""
        adapter = DhanMarketDataAdapter(
            client({"/fundlimit": json.dumps({"sodLimit": 100})}), master
        )
        with pytest.raises(DataAdapterError) as exc:
            await adapter.get_account_snapshot()
        assert "availabelBalance" in str(exc.value)
        assert "dhan_probe" in str(exc.value)


class TestBars:
    async def test_daily_candles_parse_from_parallel_arrays(self, master):
        adapter = DhanMarketDataAdapter(client({"/charts/historical": CANDLES}), master)
        bars = await adapter.get_index_bars("NIFTY", BarInterval.DAY, 10)
        assert len(bars) == 3
        assert bars[0].open == Decimal("23800.0")
        assert bars[-1].close == Decimal("23914.45")

    async def test_bars_come_back_oldest_first(self, master):
        adapter = DhanMarketDataAdapter(client({"/charts/historical": CANDLES}), master)
        bars = await adapter.get_index_bars("NIFTY", BarInterval.DAY, 10)
        assert [b.timestamp for b in bars] == sorted(b.timestamp for b in bars)

    async def test_timestamps_are_timezone_aware(self, master):
        """A naive datetime would make every session boundary and
        time-to-expiry downstream depend on the host's timezone."""
        adapter = DhanMarketDataAdapter(client({"/charts/historical": CANDLES}), master)
        bars = await adapter.get_index_bars("NIFTY", BarInterval.DAY, 10)
        assert all(bar.timestamp.tzinfo is not None for bar in bars)

    async def test_mismatched_array_lengths_raise(self, master):
        """A short volume array zipped against a long close array would
        silently mis-pair every bar after the gap — plausible-looking
        nonsense."""
        broken = json.dumps(
            {
                "open": [1, 2, 3],
                "high": [1, 2, 3],
                "low": [1, 2, 3],
                "close": [1, 2],
                "timestamp": [1787887800, 1787974200, 1788233400],
            }
        )
        adapter = DhanMarketDataAdapter(client({"/charts/historical": broken}), master)
        with pytest.raises(DataAdapterError, match="mismatched lengths"):
            await adapter.get_index_bars("NIFTY", BarInterval.DAY, 10)

    async def test_a_missing_array_raises_and_points_at_the_probe(self, master):
        adapter = DhanMarketDataAdapter(
            client({"/charts/historical": json.dumps({"close": [1]})}), master
        )
        with pytest.raises(DataAdapterError) as exc:
            await adapter.get_index_bars("NIFTY", BarInterval.DAY, 10)
        assert "dhan_probe" in str(exc.value)

    async def test_an_unsupported_interval_is_refused(self, master):
        adapter = DhanMarketDataAdapter(client({}), master)
        with pytest.raises(DataAdapterError, match="does not serve"):
            await adapter.get_index_bars("NIFTY", "4h", 10)  # type: ignore[arg-type]

    async def test_the_request_names_the_index_segment(self, master):
        """A wrong segment code is where this kind of integration goes quietly
        wrong, so the body is asserted, not just the reply."""
        session = RecordedSession({"/charts/historical": CANDLES})
        adapter = DhanMarketDataAdapter(DhanClient(config(), session), master)
        await adapter.get_index_bars("NIFTY", BarInterval.DAY, 10)
        _, body = session.posted[-1]
        assert body["exchangeSegment"] == "IDX_I"
        assert body["securityId"] == "13"
        assert body["instrument"] == "INDEX"


class TestOptionChain:
    async def test_both_sides_of_a_strike_are_read(self, master):
        adapter = DhanMarketDataAdapter(client({"/optionchain": CHAIN}), master)
        chain = await adapter.get_option_chain("NIFTY", NEAR_EXPIRY)
        assert len(chain) == 3  # 23900 CE+PE, 24000 CE only

    async def test_a_one_sided_strike_is_not_invented(self, master):
        adapter = DhanMarketDataAdapter(client({"/optionchain": CHAIN}), master)
        chain = await adapter.get_option_chain("NIFTY", NEAR_EXPIRY)
        at_24000 = [q for q in chain if q.contract.strike == Decimal(24000)]
        assert len(at_24000) == 1
        assert at_24000[0].contract.option_type is OptionType.CE

    async def test_open_interest_change_is_derived_not_copied(self, master):
        """Dhan reports the previous day's OI, not the change. Reporting the
        raw previous value as a change would invert the meaning of every OI
        build."""
        adapter = DhanMarketDataAdapter(client({"/optionchain": CHAIN}), master)
        chain = await adapter.get_option_chain("NIFTY", NEAR_EXPIRY)
        ce = next(
            q
            for q in chain
            if q.contract.strike == Decimal(23900)
            and q.contract.option_type is OptionType.CE
        )
        assert ce.open_interest == 95996
        assert ce.open_interest_change == 95996 - 21027

    async def test_an_unwind_comes_through_negative(self, master):
        adapter = DhanMarketDataAdapter(client({"/optionchain": CHAIN}), master)
        chain = await adapter.get_option_chain("NIFTY", NEAR_EXPIRY)
        at_24000 = next(q for q in chain if q.contract.strike == Decimal(24000))
        assert at_24000.open_interest_change == 60000 - 75000

    async def test_greeks_are_recomputed_by_default(self, master):
        """Not Dhan's, even though it publishes them: one rate and one
        day-count convention has to apply across every provider or delta fit
        compares quantities that are not the same quantity."""
        adapter = DhanMarketDataAdapter(client({"/optionchain": CHAIN}), master)
        chain = await adapter.get_option_chain("NIFTY", NEAR_EXPIRY)
        ce = next(
            q
            for q in chain
            if q.contract.strike == Decimal(23900)
            and q.contract.option_type is OptionType.CE
        )
        assert ce.greeks is not None
        # Dhan's published delta for this leg is 0.5601. Ours will be close
        # but not identical, which is the point.
        assert ce.greeks.delta != Decimal("0.5601")
        assert Decimal("0.3") < ce.greeks.delta < Decimal("0.8")

    async def test_broker_greeks_can_be_trusted_deliberately(self, master):
        adapter = DhanMarketDataAdapter(
            client({"/optionchain": CHAIN}, trust_broker_greeks=True), master
        )
        chain = await adapter.get_option_chain("NIFTY", NEAR_EXPIRY)
        ce = next(
            q
            for q in chain
            if q.contract.strike == Decimal(23900)
            and q.contract.option_type is OptionType.CE
        )
        assert ce.greeks is not None
        assert ce.greeks.delta == Decimal("0.5601")

    async def test_lot_size_comes_from_the_instrument_master(self, master):
        """Not from the chain response. The master is the exchange's record,
        and it is what caught the 75-versus-65 error."""
        adapter = DhanMarketDataAdapter(client({"/optionchain": CHAIN}), master)
        chain = await adapter.get_option_chain("NIFTY", NEAR_EXPIRY)
        assert all(q.contract.lot_size == 65 for q in chain)

    async def test_an_empty_chain_raises(self, master):
        body = json.dumps({"data": {"last_price": 1, "oc": {}}})
        adapter = DhanMarketDataAdapter(client({"/optionchain": body}), master)
        with pytest.raises(DataAdapterError, match="empty chain"):
            await adapter.get_option_chain("NIFTY", NEAR_EXPIRY)

    async def test_expiries_come_from_the_master_not_an_api_call(self, master):
        """Spending a rate-limited call to learn something already on disk
        would be worse in every respect, and it could disagree with the lot
        sizes read from the same file."""
        session = RecordedSession({})
        adapter = DhanMarketDataAdapter(DhanClient(config(), session), master)
        expiries = await adapter.get_available_expiries("NIFTY")
        assert expiries[0] == NEAR_EXPIRY
        assert session.requests == []


class TestIndexQuote:
    async def test_a_flat_ohlc_payload_is_read(self, master):
        body = json.dumps(
            {
                "data": {
                    "IDX_I": {
                        "13": {
                            "last_price": 23914.45,
                            "open": 23858.0,
                            "high": 23930.0,
                            "low": 23786.8,
                            "close": 24055.8,
                        }
                    }
                }
            }
        )
        adapter = DhanMarketDataAdapter(client({"/marketfeed/ohlc": body}), master)
        quote = await adapter.get_index_quote("NIFTY")
        assert quote.ltp == Decimal("23914.45")
        assert quote.previous_close == Decimal("24055.8")

    async def test_a_nested_ohlc_payload_is_also_read(self, master):
        """Dhan nests OHLC on some feed endpoints and flattens it on others.
        Accepting both avoids a KeyError that reads as "the index has no open
        price"."""
        body = json.dumps(
            {
                "data": {
                    "IDX_I": {
                        "13": {
                            "last_price": 23914.45,
                            "ohlc": {
                                "open": 23858.0,
                                "high": 23930.0,
                                "low": 23786.8,
                                "close": 24055.8,
                            },
                        }
                    }
                }
            }
        )
        adapter = DhanMarketDataAdapter(client({"/marketfeed/ohlc": body}), master)
        assert (await adapter.get_index_quote("NIFTY")).open == Decimal("23858.0")

    async def test_vwap_is_none_because_dhan_does_not_publish_it(self, master):
        body = json.dumps(
            {"data": {"IDX_I": {"13": {"last_price": 1, "open": 1, "high": 1, "low": 1, "close": 1}}}}
        )
        adapter = DhanMarketDataAdapter(client({"/marketfeed/ohlc": body}), master)
        assert (await adapter.get_index_quote("NIFTY")).vwap is None

    async def test_a_missing_row_raises_rather_than_defaulting(self, master):
        body = json.dumps({"data": {"IDX_I": {}}})
        adapter = DhanMarketDataAdapter(client({"/marketfeed/ohlc": body}), master)
        with pytest.raises(DataAdapterError, match="no row for NIFTY"):
            await adapter.get_index_quote("NIFTY")


class TestIndexSpec:
    async def test_the_spec_is_built_from_the_master(self, master):
        adapter = DhanMarketDataAdapter(client({}), master)
        spec = await adapter.get_index_spec("NIFTY")
        assert spec.lot_size == 65
        assert spec.tick_size == Decimal("0.05")
        assert spec.strike_step == Decimal(50)


class TestBrokerAdapter:
    def order(self, **kwargs) -> OrderRequest:
        defaults = {
            "decision_id": "d1",
            "thesis_id": "t1",
            "contract": None,
            "side": OrderSide.BUY,
            "quantity": 65,
            "lots": 1,
            "limit_price": Decimal("131.60"),
            "sequence": 0,
        }
        defaults.update(kwargs)
        return OrderRequest(**defaults)  # type: ignore[arg-type]

    def contract_for(self, master, option_type=OptionType.CE):
        record = next(
            r
            for r in master.options_for("NIFTY")
            if r.option_type is option_type and r.expiry == NEAR_EXPIRY
        )
        from index_option_brain.contracts.instruments import OptionContractSpec

        return OptionContractSpec(
            underlying_symbol="NIFTY",
            expiry=record.expiry,
            strike=record.strike,
            option_type=record.option_type,
            lot_size=record.lot_size,
            tick_size=record.tick_size,
        ), record.security_id

    async def test_it_is_sandbox_by_default(self, master):
        broker = DhanBrokerAdapter(DhanClient(DhanConfig("C", "t")), master)
        assert broker.is_sandbox

    async def test_placing_an_order_resolves_the_security_id(self, master):
        """A broker takes a security id, not a strike and an expiry."""
        contract, security_id = self.contract_for(master)
        session = RecordedSession({"/orders": json.dumps({"orderId": "B1", "orderStatus": "TRANSIT"})})
        broker = DhanBrokerAdapter(DhanClient(config(), session), master)
        await broker.place_order(self.order(contract=contract))
        _, body = session.posted[-1]
        assert body["securityId"] == security_id
        assert body["exchangeSegment"] == "NSE_FNO"
        assert body["transactionType"] == "BUY"
        assert body["quantity"] == 65

    async def test_the_correlation_id_is_the_managers_idempotency_key(self, master):
        """So both ends agree on what "the same order" means and a retry
        cannot double the position."""
        contract, _ = self.contract_for(master)
        session = RecordedSession({"/orders": json.dumps({"orderId": "B1", "orderStatus": "TRANSIT"})})
        broker = DhanBrokerAdapter(DhanClient(config(), session), master)
        request = self.order(contract=contract)
        await broker.place_order(request)
        _, body = session.posted[-1]
        assert body["correlationId"] == request.client_order_id

    async def test_an_unlisted_contract_is_refused_not_guessed(self, master):
        """Guessing an id, or sending a strike as though it were one, would
        place an order on some other instrument entirely."""
        from index_option_brain.contracts.instruments import OptionContractSpec

        phantom = OptionContractSpec(
            underlying_symbol="NIFTY",
            expiry=NEAR_EXPIRY,
            strike=Decimal(99999),
            option_type=OptionType.CE,
            lot_size=65,
            tick_size=Decimal("0.05"),
        )
        broker = DhanBrokerAdapter(DhanClient(config(), RecordedSession({})), master)
        with pytest.raises(DataAdapterError, match="no security id"):
            await broker.place_order(self.order(contract=phantom))

    async def test_a_market_order_is_sent_when_there_is_no_limit(self, master):
        contract, _ = self.contract_for(master)
        session = RecordedSession({"/orders": json.dumps({"orderId": "B1", "orderStatus": "TRANSIT"})})
        broker = DhanBrokerAdapter(DhanClient(config(), session), master)
        await broker.place_order(self.order(contract=contract, limit_price=None))
        _, body = session.posted[-1]
        assert body["orderType"] == "MARKET"

    async def test_cancelling_uses_a_real_delete(self, master):
        """A GET to the same path returns the order's status. Sending one and
        treating the reply as a cancel would report success while the order
        stayed live."""
        session = RecordedSession(
            {"/orders/B1": json.dumps({"orderId": "B1", "orderStatus": "CANCELLED"})}
        )
        contract, _ = self.contract_for(master)
        placed = RecordedSession(
            {"/orders": json.dumps({"orderId": "B1", "orderStatus": "PENDING"})}
        )
        broker_for_place = DhanBrokerAdapter(DhanClient(config(), placed), master)
        known = await broker_for_place.place_order(self.order(contract=contract))

        broker = DhanBrokerAdapter(DhanClient(config(), session), master)
        result = await broker.cancel_order("B1", known=known)
        assert result.state is OrderLifecycleState.CANCELLED
        assert session.deleted == [f"{LIVE_BASE}/orders/B1"]
        # The caller's copy stays authoritative for the instrument.
        assert result.contract.strike == contract.strike
        assert result.thesis_id == "t1"

    async def test_a_reply_naming_no_instrument_is_refused_without_a_copy(self, master):
        """Inventing a contract here would put a fabricated instrument into a
        reconciled position."""
        session = RecordedSession(
            {"/orders/B1": json.dumps({"orderId": "B1", "orderStatus": "CANCELLED"})}
        )
        broker = DhanBrokerAdapter(DhanClient(config(), session), master)
        with pytest.raises(DataAdapterError, match="identifies no instrument"):
            await broker.cancel_order("B1")

    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            ("TRANSIT", OrderLifecycleState.SUBMITTED),
            ("PENDING", OrderLifecycleState.OPEN),
            ("PART_TRADED", OrderLifecycleState.PARTIAL),
            ("TRADED", OrderLifecycleState.FILLED),
            ("REJECTED", OrderLifecycleState.REJECTED),
            ("CANCELLED", OrderLifecycleState.CANCELLED),
            ("EXPIRED", OrderLifecycleState.CANCELLED),
        ],
    )
    async def test_dhan_statuses_map_onto_the_state_machine(
        self, master, status, expected
    ):
        contract, _ = self.contract_for(master)
        session = RecordedSession(
            {"/orders": json.dumps({"orderId": "B1", "orderStatus": status})}
        )
        broker = DhanBrokerAdapter(DhanClient(config(), session), master)
        order = await broker.place_order(self.order(contract=contract))
        assert order.state is expected

    async def test_an_unknown_status_is_refused_not_guessed(self, master):
        """Mapping an unrecognized status onto OPEN would leave the Order
        Manager believing a possibly-filled order is still working."""
        contract, _ = self.contract_for(master)
        session = RecordedSession(
            {"/orders": json.dumps({"orderId": "B1", "orderStatus": "SOMETHING_NEW"})}
        )
        broker = DhanBrokerAdapter(DhanClient(config(), session), master)
        with pytest.raises(DataAdapterError, match="unrecognized order status"):
            await broker.place_order(self.order(contract=contract))

    async def test_a_status_reply_reconstructs_its_contract_from_the_master(self, master):
        contract, security_id = self.contract_for(master)
        session = RecordedSession(
            {
                "/orders/B1": json.dumps(
                    {
                        "orderId": "B1",
                        "orderStatus": "TRADED",
                        "securityId": security_id,
                        "transactionType": "SELL",
                        "quantity": 65,
                        "filledQty": 65,
                        "averageTradedPrice": 131.55,
                    }
                )
            }
        )
        broker = DhanBrokerAdapter(DhanClient(config(), session), master)
        order = await broker.get_order_status("B1")
        assert order.state is OrderLifecycleState.FILLED
        assert order.filled_quantity == 65
        assert order.average_fill_price == Decimal("131.55")
        assert order.contract.strike == contract.strike

    async def test_a_status_on_an_unknown_security_id_raises(self, master):
        session = RecordedSession(
            {"/orders/B1": json.dumps({"orderId": "B1", "orderStatus": "TRADED", "securityId": "999999"})}
        )
        broker = DhanBrokerAdapter(DhanClient(config(), session), master)
        with pytest.raises(DataAdapterError, match="not in the instrument master"):
            await broker.get_order_status("B1")

    async def test_a_single_element_list_reply_is_unwrapped(self, master):
        _, security_id = self.contract_for(master)
        session = RecordedSession(
            {
                "/orders/B1": json.dumps(
                    [{"orderId": "B1", "orderStatus": "OPEN", "securityId": security_id}]
                )
            }
        )
        broker = DhanBrokerAdapter(DhanClient(config(), session), master)
        assert (await broker.get_order_status("B1")).state is OrderLifecycleState.OPEN
