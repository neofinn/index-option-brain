"""Replay NIFTY with real option chains and report P&L net of costs.

    python scripts/replay_with_chains.py --sessions 400 --hold 5

Fills are at settlement prices, which nobody trades at, so the result is
optimistic by construction: read a loss as conclusive and a profit as a
reason to go measure real spreads.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import date
from pathlib import Path

from index_option_brain.backtest.chain_replay import ChainReplayEngine
from index_option_brain.data.adapters.base import DataAdapterError
from index_option_brain.data.adapters.nse_archive import NseArchiveAdapter
from index_option_brain.data.adapters.nse_fo_archive import (
    EARLIEST_UDIFF_SESSION,
    NseFoArchiveAdapter,
)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="NIFTY")
    parser.add_argument("--sessions", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--hold", type=int, default=5)
    parser.add_argument("--cache", default="var/bhavcopy")
    args = parser.parse_args()

    archive = NseArchiveAdapter()
    try:
        series = await archive.get_many_index_bars(
            [args.symbol, "INDIAVIX"], count=args.sessions
        )
    finally:
        await archive.aclose()
    bars, vix = series[args.symbol], series["INDIAVIX"]
    days = [b.timestamp.date() for b in bars]
    usable = [d for d in days if d >= EARLIEST_UDIFF_SESSION]
    print(
        f"{len(bars)} index sessions {days[0]} -> {days[-1]}; "
        f"{len(usable)} within the UDiFF bhavcopy era"
    )

    fo = NseFoArchiveAdapter(cache_dir=Path(args.cache))
    chains: dict[date, object] = {}
    prices: dict[date, object] = {}
    unresolved: list[date] = []
    try:
        for n, day in enumerate(usable, 1):
            if n % 25 == 0 or n == len(usable):
                print(f"  chains {n}/{len(usable)}  ({len(chains)} loaded)", flush=True)
            try:
                chain = await fo.get_option_chain(args.symbol, day)
                price_map = await fo.get_price_map(args.symbol, day)
            except DataAdapterError as exc:
                unresolved.append(day)
                print(f"  ! {day}: {exc}", flush=True)
                continue
            if chain is not None:
                chains[day] = chain
            if price_map is not None:
                prices[day] = price_map
    finally:
        await fo.aclose()

    print(f"chains {len(chains)}   price maps {len(prices)}   unresolved {len(unresolved)}")
    if unresolved:
        print(f"  unresolved sessions: {[d.isoformat() for d in unresolved[:10]]}")

    engine = ChainReplayEngine(
        chains=chains,  # type: ignore[arg-type]
        prices=prices,  # type: ignore[arg-type]
        warmup=args.warmup,
        hold_sessions=args.hold,
    )
    report = engine.run(args.symbol, bars, vix_bars=vix)

    print(f"\n--- {args.symbol}, hold {args.hold} sessions, fills at settlement ---")
    print(f"sessions replayed     {report.sessions}")
    print(f"decisions taken       {report.decisions}")
    print(f"tradeable structures  {report.tradeable}")
    print(f"skipped, position on  {report.skipped_position_open}")
    print(f"unpriceable exits     {report.unpriceable_exits}")
    print("\nstrategies selected:")
    for name, count in sorted(report.strategies.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<24} {count}")

    if not report.trades:
        print("\nNo trade was placed. That is a result, not an error: with a real")
        print("chain loaded, the engine still declined every session.")
        return

    print(f"\ntrades                {len(report.trades)}")
    print(f"hit rate              {report.hit_rate:.1%}" if report.hit_rate is not None else "")
    print(f"gross                 Rs {report.gross:,.0f}")
    print(f"costs                 Rs {report.costs:,.0f}")
    print(f"net                   Rs {report.net:,.0f}")
    share = report.cost_share_of_gross
    if share is not None:
        print(f"costs / |gross|       {share:.1%}")
    best = max(report.trades, key=lambda t: t.net)
    worst = min(report.trades, key=lambda t: t.net)
    print(f"best trade            Rs {best.net:,.0f}  ({best.strategy} {best.entry_day})")
    print(f"worst trade           Rs {worst.net:,.0f}  ({worst.strategy} {worst.entry_day})")
    print("\nRemember: settlement fills. A loss here is conclusive; a profit is not.")


if __name__ == "__main__":
    asyncio.run(main())
