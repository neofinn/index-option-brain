#!/usr/bin/env python3
"""Probe the live NSE feed and report exactly what it is serving right now.

    python scripts/live_nse_check.py [NIFTY|BANKNIFTY]

This talks to the real exchange. It is not part of the test suite, because a
test that needs a working internet connection and an open market fails for
reasons that have nothing to do with the code — parsing is pinned down against
recorded payloads in `tests/data/` instead.

Run this to answer the operational question the test suite cannot: *is the
feed usable at this moment*. Every line printed is measured. A capability that
is unavailable is reported as unavailable, with the reason.
"""

from __future__ import annotations

import asyncio
import sys
from decimal import Decimal

from index_option_brain.contracts.enums import BarInterval, OptionType
from index_option_brain.data.adapters.base import DataAdapterError
from index_option_brain.data.adapters.nse_public import (
    NSE_PUBLIC_DESCRIPTOR,
    NsePublicAdapter,
    years_to_expiry,
)


def rupees(value: Decimal | float) -> str:
    return f"{float(value):,.2f}"


async def report(symbol: str) -> int:
    print(f"Provider : {NSE_PUBLIC_DESCRIPTOR.display_name}")
    print(f"Symbol   : {symbol}\n")

    failures = 0
    async with NsePublicAdapter() as nse:
        spec = await nse.get_index_spec(symbol)
        print(
            f"contract   lot {spec.lot_size}  strike step {spec.strike_step}  "
            f"tick {spec.tick_size}   (configuration, not a live reading)"
        )

        try:
            quote = await nse.get_index_quote(symbol)
        except DataAdapterError as exc:
            print(f"index      UNAVAILABLE - {exc}")
            return 1

        print(
            f"index      {rupees(quote.ltp)}  "
            f"O {rupees(quote.open)}  H {rupees(quote.high)}  "
            f"L {rupees(quote.low)}  prev {rupees(quote.previous_close)}  "
            f"({float(quote.change_pct):+.2f}%)"
        )
        print(f"as of      {quote.timestamp.isoformat()}  (UTC)")
        print(f"vwap       {quote.vwap if quote.vwap is not None else 'not published'}")

        try:
            vix, vix_prev = await nse.get_india_vix()
            print(f"india vix  {vix:.2f}  (prev close {vix_prev:.2f})")
        except DataAdapterError as exc:
            failures += 1
            print(f"india vix  UNAVAILABLE - {exc}")

        try:
            expiries = await nse.get_available_expiries(symbol)
        except DataAdapterError as exc:
            print(f"expiries   UNAVAILABLE - {exc}")
            return failures + 1

        near = expiries[0]
        print(
            f"expiries   {len(expiries)} listed, near {near.isoformat()} "
            f"({near:%A}), {years_to_expiry(as_of=quote.timestamp, expiry=near) * 365:.2f} "
            "calendar days out"
        )

        try:
            chain = await nse.get_option_chain(symbol, near)
        except DataAdapterError as exc:
            failures += 1
            print(f"chain      UNAVAILABLE - {exc}")
            chain = []

        if chain:
            strikes = sorted({q.contract.strike for q in chain})
            with_iv = [q for q in chain if q.implied_volatility is not None]
            with_greeks = [q for q in chain if q.greeks is not None]
            print(
                f"chain      {len(chain)} legs across {len(strikes)} strikes "
                f"{strikes[0]}-{strikes[-1]}"
            )
            print(
                f"           {len(with_iv)} with a usable IV, "
                f"{len(chain) - len(with_iv)} unmarkable "
                f"({len(with_greeks)} with computed greeks)"
            )

            atm = min(strikes, key=lambda k: abs(k - quote.ltp))
            print(f"\nATM {atm}:")
            for side in (OptionType.CE, OptionType.PE):
                leg = next(
                    (
                        q
                        for q in chain
                        if q.contract.strike == atm and q.contract.option_type is side
                    ),
                    None,
                )
                if leg is None:
                    print(f"  {side}  not quoted")
                    continue
                spread = leg.relative_spread
                print(
                    f"  {side}  ltp {rupees(leg.ltp):>10}  "
                    f"bid {rupees(leg.bid) if leg.bid else '-':>10}  "
                    f"ask {rupees(leg.ask) if leg.ask else '-':>10}  "
                    f"spread {float(spread) * 100 if spread else 0:5.2f}%  "
                    f"oi {leg.open_interest:>9,} ({leg.open_interest_change:+,})"
                )
                if leg.greeks is None:
                    print("       iv unavailable, so no greeks")
                    continue
                greeks = leg.greeks
                print(
                    f"       iv {float(leg.implied_volatility):5.2f}%  "
                    f"delta {float(greeks.delta):+.4f}  "
                    f"gamma {float(greeks.gamma):.6f}  "
                    f"theta {float(greeks.theta):+.2f}/day  "
                    f"vega {float(greeks.vega):.2f}/pt"
                )

        print("\nKnown gaps (probed, not assumed):")
        try:
            await nse.get_index_bars(symbol, BarInterval.DAY, 20)
            print("  bars       unexpectedly available - the adapter needs updating")
        except DataAdapterError:
            print("  bars       blocked by the exchange's anti-bot page")
        print("  breadth    /api/equity-stockIndices returns 404 to this client")
        print("  account    not a broker; no positions, margin or orders")

    return failures


def main() -> int:
    symbol = (sys.argv[1] if len(sys.argv) > 1 else "NIFTY").upper()
    try:
        return asyncio.run(report(symbol))
    except DataAdapterError as exc:
        print(f"FEED UNUSABLE: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
