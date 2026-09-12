"""Order placement on Delta Exchange India.

Verification status, stated up front because it decides how this may be used
-----------------------------------------------------------------------------
**Request shapes are verified.** Every field name, type and enum value below
was read out of `delta_rest_client`'s own source — the client builds the
payload itself, so `product_id`, `size`, `side`, `order_type`,
`limit_price`, `time_in_force`, `client_order_id`, `post_only` and
`reduce_only` are the library's actual keys, not an inference from
documentation.

**Response shapes are NOT verified.** Placing an order needs a key, and no
key has been used against this venue. So every response mapping here is an
inference, and the adapter is built to fail loudly rather than plausibly:
an unrecognised order state raises instead of being mapped to something
that looks reasonable. `verified_capabilities` is empty and says so.

That distinction is the whole reason this file is separate from the data
adapter. The data side was verified against live public endpoints before it
was written; this side cannot be, yet.

Two things Delta gives the execution layer that Dhan does not
-------------------------------------------------------------
* **`client_order_id`.** A caller-supplied idempotency key, honoured by the
  venue. The Execution Gate's duplicate-order check has been relying on a
  working-memory store that does not exist yet; a broker-enforced key is
  stronger than any local one, because it survives a crash between the
  send and the record.
* **`reduce_only`.** An exit can be marked as such, so a closing order
  cannot accidentally open a new position on the other side — which is what
  a stale quantity does at the worst possible moment.

Units
-----
Delta's `size` is a count of **contracts**, and one contract is
`contract_value` units of the underlying (0.001 BTC on the options
checked). `OrderRequest` carries both `lots` and `quantity` (units) so the
two can never be silently confused, and this adapter sends `lots` —
asserting the relationship rather than trusting it, because the Indian
convention maps the other way and a wrong choice here is a position off by
three orders of magnitude.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from functools import partial
from typing import Any, Protocol

from index_option_brain.contracts.enums import OrderLifecycleState, OrderSide
from index_option_brain.contracts.order import Order, OrderRequest
from index_option_brain.data.adapters.base import DataAdapterError
from index_option_brain.data.adapters.delta_exchange import (
    TESTNET_INDIA,
    parse_option_symbol,
)
from index_option_brain.execution.broker_adapter import BrokerAdapter

#: Delta's own enum values, read from delta_rest_client's source.
LIMIT_ORDER = "limit_order"
MARKET_ORDER = "market_order"
TIME_IN_FORCE_IOC = "ioc"
TIME_IN_FORCE_GTC = "gtc"

#: Inferred, NOT verified. Every entry here is a guess about a payload no
#: one has seen, which is why `_lifecycle` raises on anything absent rather
#: than choosing a default: a broker state silently mapped to OPEN when it
#: meant REJECTED is a position the system believes it has and does not.
_INFERRED_STATES: dict[str, OrderLifecycleState] = {
    "open": OrderLifecycleState.OPEN,
    "pending": OrderLifecycleState.SUBMITTED,
    "closed": OrderLifecycleState.FILLED,
    "cancelled": OrderLifecycleState.CANCELLED,
    "canceled": OrderLifecycleState.CANCELLED,
    "partially_filled": OrderLifecycleState.PARTIAL,
    "rejected": OrderLifecycleState.REJECTED,
}


class DeltaOrderClient(Protocol):
    """The slice of `DeltaRestClient` this adapter uses."""

    def get_products(
        self, query: dict[str, Any] | None = ..., auth: bool = ...
    ) -> Any: ...

    def place_order(self, **kwargs: Any) -> Any: ...

    def cancel_order(self, product_id: int, order_id: int) -> Any: ...

    def get_order_by_id(self, order_id: int) -> Any: ...

    def get_order_by_client_id(self, client_oid: str) -> Any: ...

    def get_all_wallet_balances(self) -> Any: ...


def _decimal(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _int(value: Any) -> int | None:
    resolved = _decimal(value)
    return None if resolved is None else int(resolved)


def client_order_id(request: OrderRequest) -> str:
    """Delta's form of `OrderRequest.client_order_id`.

    The contract already derives a stable per-leg id from the decision and
    the sequence, which is what makes resubmission idempotent — a retry
    after a timeout carries the same key, so the venue rejects the duplicate
    instead of opening a second position. That id is reused rather than
    replaced: a parallel scheme here would mean the Order Manager and the
    broker disagreed about which order is which.

    Only the separator changes. Delta's accepted charset and length for this
    field have not been observed, so ':' is replaced with '-' and the result
    truncated — conservative choices, and both unverified like everything
    else on the response side.
    """
    return request.client_order_id.replace(":", "-")[:64]


@dataclass
class DeltaBrokerAdapter(BrokerAdapter):
    """Places orders on Delta Exchange India.

    `implemented` and `verified_capabilities` are kept apart deliberately.
    Every method below is implemented; none is verified, because that needs
    a key. A caller that treats "implemented" as "safe to trade" has made
    the mistake this separation exists to prevent.
    """

    client: DeltaOrderClient
    base_url: str = TESTNET_INDIA
    """Testnet by default. Production requires saying so."""
    dry_run: bool = True
    """When True, `place_order` refuses rather than sending.

    Default True because the response mapping is unverified. Turning it off
    is the deliberate act of accepting that — and should follow a probe
    against a testnet key, not precede it.
    """
    _product_ids: dict[str, int] = field(default_factory=dict)

    @property
    def verified_capabilities(self) -> frozenset[str]:
        """Nothing. No response from this venue has been observed."""
        return frozenset()

    async def _call(self, fn: Any, /, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, partial(fn, **kwargs))
        except Exception as exc:
            raise DataAdapterError(f"Delta order request failed: {exc}") from exc

    async def resolve_product_id(self, symbol: str) -> int:
        """Delta addresses instruments by integer id, not by symbol.

        Resolved from the live product list and cached. An unresolvable
        symbol raises: sending an order against a guessed id would trade a
        different instrument, which is the worst available failure.
        """
        key = symbol.strip().upper()
        if key in self._product_ids:
            return self._product_ids[key]

        option_type, asset, _, _ = parse_option_symbol(key)
        del option_type
        products = await self._call(
            self.client.get_products,
            query={
                "contract_types": "call_options,put_options",
                "underlying_asset_symbols": asset,
            },
            auth=False,
        )
        for product in products or []:
            if not isinstance(product, dict):
                continue
            found = str(product.get("symbol", "")).upper()
            product_id = _int(product.get("id"))
            if found and product_id is not None:
                self._product_ids[found] = product_id
        if key not in self._product_ids:
            raise DataAdapterError(
                f"Delta lists no product for {symbol!r}, so no order can be "
                "addressed to it. Refusing rather than guessing an id."
            )
        return self._product_ids[key]

    @staticmethod
    def delta_symbol(request: OrderRequest) -> str:
        """Delta's symbol for this contract: C-BTC-79800-060926.

        Rebuilt from the contract rather than carried, so a symbol and a
        strike cannot disagree.
        """
        contract = request.contract
        kind = "C" if contract.option_type.value == "CE" else "P"
        return (
            f"{kind}-{contract.underlying_symbol.upper()}-"
            f"{contract.strike:.0f}-{contract.expiry:%d%m%y}"
        )

    def _size(self, request: OrderRequest) -> int:
        """Contracts to send, with the unit relationship asserted.

        Delta counts contracts; `OrderRequest.quantity` counts units and
        `lots` counts lots. On this venue the adapter's IndexSpec reports
        lot_size 1 — the real multiplier is `contract_multiplier` on the
        quote — so the two coincide. Asserting that rather than assuming it,
        because if they ever diverge the wrong choice is a position off by
        the multiplier and the number looks plausible either way.
        """
        lot_size = request.contract.lot_size
        if request.quantity != request.lots * lot_size:
            raise DataAdapterError(
                f"OrderRequest is internally inconsistent: {request.quantity} "
                f"units is not {request.lots} lots x {lot_size}. Refusing to "
                "pick one."
            )
        if lot_size != 1:
            raise DataAdapterError(
                f"Delta sizes orders in contracts, but this contract reports "
                f"lot_size {lot_size}. The multiplier belongs on the quote's "
                "contract_multiplier, not in lot_size, and sending either "
                "number here would be a guess."
            )
        return request.lots

    async def place_order(self, request: OrderRequest) -> Order:
        if self.dry_run:
            raise DataAdapterError(
                "DeltaBrokerAdapter is in dry_run: no response from this "
                "venue has been observed, so every state mapping here is an "
                "inference. Run scripts/delta_probe.py with a testnet key, "
                "confirm the shapes, then set dry_run=False deliberately."
            )
        symbol = self.delta_symbol(request)
        product_id = await self.resolve_product_id(symbol)
        size = self._size(request)

        payload: dict[str, Any] = {
            "product_id": product_id,
            "size": size,
            "side": "buy" if request.side is OrderSide.BUY else "sell",
            # A limit order whenever a price is set. Market orders on an
            # options book with a wide spread are how a defined-risk
            # structure acquires an undefined entry.
            "order_type": LIMIT_ORDER if request.limit_price is not None else MARKET_ORDER,
            "client_order_id": client_order_id(request),
            "post_only": "false",
            # Always false. Delta supports reduce_only, which would stop a
            # closing order from accidentally opening a position on the other
            # side when a stale quantity is sent — a real safeguard. But
            # OrderRequest does not distinguish an entry from an exit, so
            # there is nothing here to set it from, and inferring it from
            # side or from portfolio state would be a guess at the one moment
            # a guess is most expensive. Wiring it needs the flag on the
            # contract first.
            "reduce_only": "false",
        }
        if request.limit_price is not None:
            payload["limit_price"] = str(request.limit_price)
            payload["time_in_force"] = TIME_IN_FORCE_GTC

        response = await self._call(self.client.place_order, **payload)
        return self._to_order(response, request=request)

    async def cancel_order(
        self, broker_order_id: str, *, known: Order | None = None
    ) -> Order:
        if known is None:
            raise DataAdapterError(
                "Delta cancels by (order_id, product_id) and its reply does "
                "not identify the instrument. Pass the caller's copy of the "
                "order: reconstructing one would put a fabricated contract "
                "into a reconciled position."
            )
        product_id = await self.resolve_product_id(
            self.delta_symbol_for(known)
        )
        order_id = _int(broker_order_id)
        if order_id is None:
            raise DataAdapterError(
                f"Delta order ids are integers; {broker_order_id!r} is not one"
            )
        response = await self._call(
            self.client.cancel_order, product_id=product_id, order_id=order_id
        )
        return self._to_order(response, known=known)

    async def get_order_status(
        self, broker_order_id: str, *, known: Order | None = None
    ) -> Order:
        order_id = _int(broker_order_id)
        if order_id is None:
            raise DataAdapterError(
                f"Delta order ids are integers; {broker_order_id!r} is not one"
            )
        response = await self._call(self.client.get_order_by_id, order_id=order_id)
        return self._to_order(response, known=known)

    @staticmethod
    def delta_symbol_for(order: Order) -> str:
        contract = order.contract
        kind = "C" if contract.option_type.value == "CE" else "P"
        return (
            f"{kind}-{contract.underlying_symbol.upper()}-"
            f"{contract.strike:.0f}-{contract.expiry:%d%m%y}"
        )

    @staticmethod
    def _lifecycle(raw: Any) -> OrderLifecycleState:
        """Map a Delta order state, or raise.

        Raises on anything unrecognised rather than defaulting. A broker
        state quietly mapped to OPEN when it meant REJECTED is a position
        the system believes it holds and does not — and since none of these
        mappings has been observed against a live response, the unknown case
        is the likely one rather than the exotic one.
        """
        key = str(raw or "").strip().lower()
        state = _INFERRED_STATES.get(key)
        if state is None:
            raise DataAdapterError(
                f"Delta returned order state {raw!r}, which this adapter has "
                f"no mapping for. Known: {sorted(_INFERRED_STATES)}. Refusing "
                "to guess a lifecycle state."
            )
        return state

    def _to_order(
        self,
        response: Any,
        *,
        request: OrderRequest | None = None,
        known: Order | None = None,
    ) -> Order:
        """Build an Order from a reply, using the caller's copy for identity.

        The contract, decision and thesis come from the request or the known
        order — never from the reply. A broker reply is authoritative about
        *state and fills* and about nothing else; letting it supply the
        instrument is how a fabricated contract enters the system.
        """
        if request is None and known is None:
            raise DataAdapterError(
                "Cannot build an Order without either the request that "
                "created it or the caller's copy"
            )
        body = response if isinstance(response, dict) else {}
        if isinstance(body.get("result"), dict):
            body = body["result"]

        broker_id = body.get("id")
        state = self._lifecycle(body.get("state"))
        filled = _int(body.get("size")) or 0
        unfilled = _int(body.get("unfilled_size"))
        if unfilled is not None:
            filled = max(filled - unfilled, 0)

        now = datetime.now(UTC)
        if known is not None:
            return known.model_copy(
                update={
                    "state": state,
                    "broker_order_id": str(broker_id) if broker_id else known.broker_order_id,
                    "filled_quantity": filled or known.filled_quantity,
                    "average_fill_price": _decimal(body.get("average_fill_price"))
                    or known.average_fill_price,
                    "updated_at": now,
                }
            )

        assert request is not None
        return Order(
            order_id=client_order_id(request),
            decision_id=request.decision_id,
            thesis_id=request.thesis_id,
            contract=request.contract,
            side=request.side,
            quantity=request.quantity,
            limit_price=request.limit_price,
            state=state,
            broker_order_id=str(broker_id) if broker_id else None,
            filled_quantity=filled,
            average_fill_price=_decimal(body.get("average_fill_price")),
            created_at=now,
            updated_at=now,
        )
