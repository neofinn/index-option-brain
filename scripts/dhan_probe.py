#!/usr/bin/env python3
"""Capture Dhan's real response shapes, so the field mapping stops guessing.

    DHAN_CLIENT_ID=... DHAN_ACCESS_TOKEN=... python scripts/dhan_probe.py
    DHAN_SANDBOX=false ... python scripts/dhan_probe.py     # live host

Why this exists
---------------
The Dhan adapter's routes and error envelopes were verified against the live
hosts without credentials. Its **response bodies** were not — those need a
token, and the field mapping follows published documentation until one
exists.

The NSE adapter is the reason that distinction is taken seriously. There, the
documented `bidprice` field was always null and the real top of book lived in
`buyPrice1`; an adapter written from the documentation alone would have
produced a chain with no bid or ask anywhere and no error to show for it.

So run this once. It prints the actual keys of every response, flags the ones
the adapter expects and did not find, and then runs the adapter itself so a
mismatch surfaces as a precise error rather than as a silent zero in a sizing
calculation.

It is read-only. It places no order, and it defaults to the sandbox.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

from index_option_brain.contracts.enums import BarInterval
from index_option_brain.data.adapters.base import DataAdapterError
from index_option_brain.data.adapters.dhan import (
    INDEX_SEGMENT,
    DhanClient,
    DhanConfig,
    DhanMarketDataAdapter,
)
from index_option_brain.data.dhan_instruments import DhanInstrumentMaster

EXPECTED_FIELDS: dict[str, tuple[str, ...]] = {
    "/fundlimit": ("availabelBalance", "utilizedAmount", "sodLimit"),
    "/charts/historical": ("open", "high", "low", "close", "timestamp"),
    "/charts/intraday": ("open", "high", "low", "close", "timestamp"),
    "/marketfeed/ohlc": ("data",),
    "/optionchain": ("data",),
}


def describe(value: Any, depth: int = 0) -> str:
    """A shape, not a dump. Real payloads are long and the keys are the point."""
    pad = "  " * depth
    if isinstance(value, dict):
        if depth >= 2:
            return f"{{{len(value)} keys: {', '.join(list(value)[:8])}}}"
        lines = []
        for key, item in list(value.items())[:14]:
            lines.append(f"{pad}  {key}: {describe(item, depth + 1)}")
        more = "" if len(value) <= 14 else f"\n{pad}  ... {len(value) - 14} more"
        return "{\n" + "\n".join(lines) + more + f"\n{pad}}}"
    if isinstance(value, list):
        if not value:
            return "[] (empty)"
        return f"[{len(value)} x {describe(value[0], depth + 1)}]"
    if isinstance(value, str):
        return f"str {value[:40]!r}"
    return f"{type(value).__name__} {value!r}"


def check_expected(path: str, payload: Any) -> list[str]:
    expected = EXPECTED_FIELDS.get(path, ())
    if not expected or not isinstance(payload, dict):
        return []
    return [field for field in expected if field not in payload]


async def probe(client: DhanClient, method: str, path: str, body: Any = None) -> Any:
    print(f"\n{'=' * 72}\n{method} {path}")
    try:
        payload = await (
            client.get(path) if method == "GET" else client.post(path, body or {})
        )
    except DataAdapterError as exc:
        print(f"  FAILED: {exc}")
        return None
    print(describe(payload))
    missing = check_expected(path, payload)
    if missing:
        # The whole reason for the script.
        print(f"\n  !! ADAPTER EXPECTS THESE AND THEY ARE ABSENT: {missing}")
        print("     Correct the mapping in data/adapters/dhan.py before trusting it.")
    return payload


async def main() -> int:
    client_id = os.environ.get("DHAN_CLIENT_ID")
    access_token = os.environ.get("DHAN_ACCESS_TOKEN")
    if not client_id or not access_token:
        print(
            "Set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN.\n"
            "Generate them in the Dhan web console; the token is long-lived.",
            file=sys.stderr,
        )
        return 2

    sandbox = os.environ.get("DHAN_SANDBOX", "true").lower() != "false"
    config = DhanConfig(
        client_id=client_id, access_token=access_token, sandbox=sandbox
    )
    print(f"Host      : {config.base_url}")
    print(f"Mode      : {'SANDBOX' if sandbox else 'LIVE — real account'}")

    master = await DhanInstrumentMaster.load()
    security_id = master.index_security_id("NIFTY")
    expiry = master.expiries("NIFTY")[0]
    print(f"Instruments: NIFTY id={security_id} lot={master.lot_size('NIFTY')} "
          f"near expiry={expiry} ({expiry:%A})")

    async with DhanClient(config) as client:
        from datetime import datetime, timedelta

        from index_option_brain.data.adapters.dhan import IST

        today = datetime.now(IST).date()

        await probe(client, "GET", "/fundlimit")
        await probe(client, "POST", "/marketfeed/ohlc", {INDEX_SEGMENT: [int(security_id)]})
        await probe(
            client,
            "POST",
            "/charts/historical",
            {
                "securityId": security_id,
                "exchangeSegment": INDEX_SEGMENT,
                "instrument": "INDEX",
                "fromDate": (today - timedelta(days=30)).isoformat(),
                "toDate": today.isoformat(),
            },
        )
        await probe(
            client,
            "POST",
            "/charts/intraday",
            {
                "securityId": security_id,
                "exchangeSegment": INDEX_SEGMENT,
                "instrument": "INDEX",
                "interval": "5",
                "fromDate": (today - timedelta(days=5)).isoformat(),
                "toDate": today.isoformat(),
            },
        )
        await probe(
            client,
            "POST",
            "/optionchain/expirylist",
            {"UnderlyingScrip": int(security_id), "UnderlyingSeg": INDEX_SEGMENT},
        )
        await probe(
            client,
            "POST",
            "/optionchain",
            {
                "UnderlyingScrip": int(security_id),
                "UnderlyingSeg": INDEX_SEGMENT,
                "Expiry": expiry.isoformat(),
            },
        )

        # Now the adapter itself. The shapes above say what Dhan sends; this
        # says whether the mapping actually reads it.
        print(f"\n{'=' * 72}\nADAPTER END TO END")
        adapter = DhanMarketDataAdapter(client, master)
        for label, call in (
            ("index spec", lambda: adapter.get_index_spec("NIFTY")),
            ("index quote", lambda: adapter.get_index_quote("NIFTY")),
            ("daily bars", lambda: adapter.get_index_bars("NIFTY", BarInterval.DAY, 20)),
            ("5m bars", lambda: adapter.get_index_bars("NIFTY", BarInterval.MINUTE_5, 20)),
            ("option chain", lambda: adapter.get_option_chain("NIFTY", expiry)),
            ("account", lambda: adapter.get_account_snapshot()),
        ):
            try:
                result = await call()
            except DataAdapterError as exc:
                print(f"  {label:13} FAILED: {exc}")
                continue
            summary = (
                f"{len(result)} item(s)" if isinstance(result, list) else repr(result)[:150]
            )
            print(f"  {label:13} OK: {summary}")

    print(
        "\nIf every line above says OK, the mapping is verified against live "
        "responses.\nUpdate the note on DHAN_DESCRIPTOR to say so, and set the "
        "capabilities\nthe probe actually proved."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
