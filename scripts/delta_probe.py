"""Verify Delta Exchange order response shapes against a real key.

The one thing standing between DeltaBrokerAdapter and live trading. Its
request payloads were read out of `delta_rest_client`'s source and are
correct; its *response* mappings are inferences about payloads nobody has
seen, and `dry_run` defaults to True because of that.

This script closes that gap. It places a deliberately unfillable limit
order on **testnet**, prints the raw response, reads it back, cancels it,
and reports whether every state it saw is one the adapter knows. Nothing
here is a test of the adapter's logic — it is a recording of what the venue
actually returns, so the mapping stops being a guess.

    DELTA_API_KEY=... DELTA_API_SECRET=... python scripts/delta_probe.py

Testnet only unless --production is passed, and it will refuse a
production run without it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from delta_rest_client import DeltaRestClient

from index_option_brain.data.adapters.delta_exchange import (
    PRODUCTION_INDIA,
    TESTNET_INDIA,
)
from index_option_brain.execution.delta_broker import _INFERRED_STATES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--underlying", default="BTC")
    parser.add_argument("--production", action="store_true")
    args = parser.parse_args()

    key, secret = os.getenv("DELTA_API_KEY"), os.getenv("DELTA_API_SECRET")
    if not key or not secret:
        print("Set DELTA_API_KEY and DELTA_API_SECRET.", file=sys.stderr)
        return 2

    base = PRODUCTION_INDIA if args.production else TESTNET_INDIA
    print(f"environment: {base}")
    client = DeltaRestClient(base_url=base, api_key=key, api_secret=secret)

    seen: set[str] = set()

    def record(label: str, payload: object) -> None:
        print(f"\n--- {label} ---")
        print(json.dumps(payload, indent=1, default=str)[:2000])
        body = payload if isinstance(payload, dict) else {}
        if isinstance(body.get("result"), dict):
            body = body["result"]
        state = body.get("state")
        if isinstance(state, str):
            seen.add(state)

    products = [
        p
        for p in client.get_products(
            query={
                "contract_types": "call_options",
                "underlying_asset_symbols": args.underlying,
            },
            auth=False,
        )
        or []
        if p.get("state") == "live"
    ]
    if not products:
        print("No live products.", file=sys.stderr)
        return 1
    product = products[0]
    print(f"product: {product['symbol']} id={product['id']}")

    record("wallet balances", client.get_all_wallet_balances())

    # A limit price far below any conceivable market, so it rests and is
    # never filled. The point is to observe the reply, not to trade.
    placed = client.place_order(
        product_id=int(product["id"]),
        size=1,
        side="buy",
        limit_price="0.1",
        client_order_id="probe-shape-check",
    )
    record("place_order", placed)

    body = placed.get("result", placed) if isinstance(placed, dict) else {}
    order_id = body.get("id")
    if order_id is not None:
        record("get_order_by_id", client.get_order_by_id(order_id=int(order_id)))
        record(
            "cancel_order",
            client.cancel_order(
                product_id=int(product["id"]), order_id=int(order_id)
            ),
        )

    print("\n=== states observed ===")
    unknown = {s for s in seen if s.strip().lower() not in _INFERRED_STATES}
    for state in sorted(seen):
        mark = "UNMAPPED" if state in unknown else "ok"
        print(f"  {state:<20} {mark}")
    if unknown:
        print(
            f"\n{len(unknown)} state(s) have no mapping in delta_broker."
            "_INFERRED_STATES. Add them before setting dry_run=False."
        )
        return 1
    print("\nEvery observed state is mapped. dry_run=False is defensible now.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
