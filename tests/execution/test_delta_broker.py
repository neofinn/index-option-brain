"""Delta broker adapter.

The point of these tests is the refusals. Request shapes here were read out
of `delta_rest_client`'s own source and are therefore verified; response
shapes were not observed at all, so what matters is that the adapter fails
loudly wherever it is guessing rather than producing something plausible.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

import pytest

from index_option_brain.contracts.enums import (
    OptionType,
    OrderLifecycleState,
    OrderSide,
)
from index_option_brain.contracts.instruments import OptionContractSpec
from index_option_brain.contracts.order import Order, OrderRequest
from index_option_brain.data.adapters.base import DataAdapterError
from index_option_brain.execution.delta_broker import (
    DeltaBrokerAdapter,
    client_order_id,
)

NOW = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)


def contract(strike: float = 79800, lot_size: int = 1) -> OptionContractSpec:
    return OptionContractSpec(
        underlying_symbol="BTC",
        expiry=date(2026, 9, 6),
        strike=Decimal(str(strike)),
        option_type=OptionType.CE,
        lot_size=lot_size,
        tick_size=Decimal("0.1"),
    )


def request(*, lots: int = 2, lot_size: int = 1, quantity: int | None = None) -> OrderRequest:
    spec = contract(lot_size=lot_size)
    return OrderRequest(
        decision_id="d-1",
        thesis_id="t-1",
        contract=spec,
        side=OrderSide.BUY,
        quantity=lots * lot_size if quantity is None else quantity,
        lots=lots,
        limit_price=Decimal("1600.5"),
        sequence=1,
    )


class FakeClient:
    def __init__(self, *, order: dict[str, Any] | None = None) -> None:
        self.order = order or {
            "id": 987654,
            "state": "open",
            "size": 2,
            "unfilled_size": 2,
        }
        self.sent: list[dict[str, Any]] = []

    def get_products(self, query: Any = None, auth: bool = False) -> Any:
        return [
            {"id": 150989, "symbol": "C-BTC-79800-060926"},
            {"id": 150990, "symbol": "P-BTC-79800-060926"},
        ]

    def place_order(self, **kwargs: Any) -> Any:
        self.sent.append(kwargs)
        return self.order

    def cancel_order(self, product_id: int, order_id: int) -> Any:
        return {**self.order, "state": "cancelled"}

    def get_order_by_id(self, order_id: int) -> Any:
        return self.order


def live(client: FakeClient) -> DeltaBrokerAdapter:
    """An adapter with dry_run deliberately off, as a caller would have to do."""
    return DeltaBrokerAdapter(client=client, dry_run=False)


class TestVerificationPosture:
    def test_nothing_is_claimed_as_verified(self) -> None:
        """No response from this venue has been observed."""
        assert DeltaBrokerAdapter(client=FakeClient()).verified_capabilities == frozenset()

    def test_dry_run_is_the_default_and_refuses_to_send(self) -> None:
        """Turning it off should be a deliberate act that follows a probe,
        not precedes it."""
        adapter = DeltaBrokerAdapter(client=FakeClient())
        assert adapter.dry_run is True

    async def test_dry_run_names_what_is_unverified(self) -> None:
        adapter = DeltaBrokerAdapter(client=FakeClient())
        with pytest.raises(DataAdapterError, match="dry_run"):
            await adapter.place_order(request())

    def test_the_default_environment_is_testnet(self) -> None:
        assert "testnet" in DeltaBrokerAdapter(client=FakeClient()).base_url.lower()


class TestIdempotency:
    def test_the_contracts_own_client_order_id_is_reused(self) -> None:
        """A parallel scheme here would mean the Order Manager and the broker
        disagreed about which order is which."""
        req = request()
        assert client_order_id(req) == req.client_order_id.replace(":", "-")

    def test_the_same_leg_produces_the_same_key_every_time(self) -> None:
        """The whole value of the field: a retry after a timeout must be
        recognised as a duplicate, not opened as a second position."""
        assert client_order_id(request()) == client_order_id(request())

    def test_different_legs_of_one_decision_do_not_collide(self) -> None:
        first = request()
        second = first.model_copy(update={"sequence": 2})
        assert client_order_id(first) != client_order_id(second)

    async def test_the_key_is_sent_with_the_order(self) -> None:
        client = FakeClient()
        await live(client).place_order(request())
        assert client.sent[0]["client_order_id"] == client_order_id(request())


class TestRequestShape:
    async def test_the_payload_uses_the_libraries_own_field_names(self) -> None:
        client = FakeClient()
        await live(client).place_order(request())
        sent = client.sent[0]
        assert set(sent) >= {
            "product_id", "size", "side", "order_type",
            "client_order_id", "post_only", "reduce_only",
        }
        assert sent["side"] == "buy"
        assert sent["order_type"] == "limit_order"
        assert sent["limit_price"] == "1600.5"

    async def test_a_priceless_request_becomes_a_market_order(self) -> None:
        client = FakeClient()
        await live(client).place_order(request().model_copy(update={"limit_price": None}))
        assert client.sent[0]["order_type"] == "market_order"
        assert "limit_price" not in client.sent[0]

    async def test_size_is_contracts_not_units(self) -> None:
        """Delta counts contracts; the Indian convention counts units, and
        the wrong choice is a position off by the multiplier."""
        client = FakeClient()
        await live(client).place_order(request(lots=2))
        assert client.sent[0]["size"] == 2

    async def test_an_inconsistent_request_is_refused(self) -> None:
        """quantity and lots disagreeing means one of them is wrong, and the
        adapter must not pick."""
        client = FakeClient()
        bad = request(lots=2, quantity=130)
        with pytest.raises(DataAdapterError, match="internally inconsistent"):
            await live(client).place_order(bad)

    async def test_a_multiplier_hidden_in_lot_size_is_refused(self) -> None:
        """On this venue the multiplier belongs on the quote. A lot_size
        other than 1 means someone put it in the wrong place, and sending
        either number would be a guess."""
        client = FakeClient()
        with pytest.raises(DataAdapterError, match="contract_multiplier"):
            await live(client).place_order(request(lots=2, lot_size=65))

    def test_the_symbol_is_rebuilt_from_the_contract(self) -> None:
        """So a symbol and a strike cannot disagree."""
        assert DeltaBrokerAdapter.delta_symbol(request()) == "C-BTC-79800-060926"


class TestProductResolution:
    async def test_a_symbol_resolves_to_the_venues_integer_id(self) -> None:
        adapter = live(FakeClient())
        assert await adapter.resolve_product_id("C-BTC-79800-060926") == 150989

    async def test_an_unlistable_symbol_is_refused_not_guessed(self) -> None:
        """Sending an order against a guessed id would trade a different
        instrument, which is the worst available failure."""
        adapter = live(FakeClient())
        with pytest.raises(DataAdapterError, match="Refusing rather than guessing"):
            await adapter.resolve_product_id("C-BTC-99999-060926")

    async def test_resolution_is_cached(self) -> None:
        adapter = live(FakeClient())
        first = await adapter.resolve_product_id("C-BTC-79800-060926")
        assert await adapter.resolve_product_id("c-btc-79800-060926") == first


class TestStateMapping:
    async def test_a_known_state_maps(self) -> None:
        client = FakeClient(order={"id": 1, "state": "open", "size": 2, "unfilled_size": 2})
        order = await live(client).place_order(request())
        assert order.state is OrderLifecycleState.OPEN
        assert order.broker_order_id == "1"

    async def test_an_unknown_state_raises_rather_than_defaulting(self) -> None:
        """A broker state quietly mapped to OPEN when it meant REJECTED is a
        position the system believes it holds and does not — and since none
        of these mappings has been observed, unknown is the likely case."""
        client = FakeClient(order={"id": 1, "state": "some_new_state", "size": 2})
        with pytest.raises(DataAdapterError, match="no mapping for"):
            await live(client).place_order(request())

    async def test_fills_are_derived_from_unfilled_size(self) -> None:
        client = FakeClient(
            order={"id": 1, "state": "partially_filled", "size": 5, "unfilled_size": 2}
        )
        order = await live(client).place_order(request(lots=5))
        assert order.filled_quantity == 3
        assert order.state is OrderLifecycleState.PARTIAL


class TestIdentityComesFromTheCaller:
    def _known(self) -> Order:
        return Order(
            order_id="d-1-1",
            decision_id="d-1",
            thesis_id="t-1",
            contract=contract(),
            side=OrderSide.BUY,
            quantity=2,
            limit_price=Decimal("1600.5"),
            state=OrderLifecycleState.OPEN,
            broker_order_id="987654",
            created_at=NOW,
            updated_at=NOW,
        )

    async def test_a_cancel_without_the_callers_copy_is_refused(self) -> None:
        """Delta's reply does not identify the instrument, and
        reconstructing one would put a fabricated contract into a reconciled
        position."""
        adapter = live(FakeClient())
        with pytest.raises(DataAdapterError, match="Pass the caller's copy"):
            await adapter.cancel_order("987654")

    async def test_a_cancel_updates_state_and_keeps_the_contract(self) -> None:
        known = self._known()
        order = await live(FakeClient()).cancel_order("987654", known=known)
        assert order.state is OrderLifecycleState.CANCELLED
        assert order.contract == known.contract
        assert order.decision_id == known.decision_id

    async def test_a_non_integer_order_id_is_refused(self) -> None:
        adapter = live(FakeClient())
        with pytest.raises(DataAdapterError, match="integers"):
            await adapter.get_order_status("not-a-number")

    async def test_status_uses_the_known_order_for_identity(self) -> None:
        known = self._known()
        order = await live(FakeClient()).get_order_status("987654", known=known)
        assert order.contract == known.contract
