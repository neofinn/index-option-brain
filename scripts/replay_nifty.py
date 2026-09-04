"""Replay the decision chain over NSE's published daily archive.

Signal quality only. NSE serves no historical option chains, so strike
selection, costs and P&L are not replayable — see backtest/replay.py.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import date

from index_option_brain.backtest.replay import DailyReplayEngine, evaluate
from index_option_brain.data.adapters.nse_archive import NseArchiveAdapter


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="NIFTY")
    parser.add_argument("--sessions", type=int, default=250)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--as-of", default=None, help="YYYY-MM-DD; defaults to today")
    args = parser.parse_args()

    as_of = date.fromisoformat(args.as_of) if args.as_of else None
    archive = NseArchiveAdapter(concurrency=6, max_unresolved=4)
    try:
        # One fetch for both series, so they cover exactly the same sessions.
        series = await archive.get_many_index_bars(
            [args.symbol, "INDIA VIX"], count=args.sessions, as_of=as_of
        )
    finally:
        await archive.aclose()

    bars = series[args.symbol]
    vix = series.get("INDIA VIX")
    if vix is not None and len(vix) != len(bars):
        # Align on the shorter tail rather than refuse: VIX history is
        # shorter in some archive stretches.
        keep = min(len(vix), len(bars))
        bars, vix = bars[-keep:], vix[-keep:]

    print(
        f"{len(bars)} sessions  {bars[0].timestamp.date()} -> {bars[-1].timestamp.date()}"
        f"   VIX series: {len(vix) if vix else 0}"
    )
    cycles = DailyReplayEngine(warmup=args.warmup).run(
        args.symbol, bars, vix_bars=vix
    )
    print(f"{len(cycles)} decisions taken\n")

    for horizon in (1, 3, 5):
        report = evaluate(cycles, horizon=horizon, all_bars=bars)
        if report.base_rate_up is None:
            continue
        print(f"--- {horizon}-session horizon ---")
        print(
            f"  base rate up {report.base_rate_up:.1%}   "
            f"mean session {report.mean_session_return:+.3f}%   "
            f"scored {report.sessions}"
        )
        for key, stats in report.by_direction.items():
            if not stats.is_measurable:
                print(f"  {key:<9} never signalled")
                continue
            hit = f"{stats.hit_rate:.1%}" if stats.hit_rate is not None else "n/a"
            se = stats.hit_rate_standard_error
            err = f" +/-{se * 100:.1f}" if se is not None else ""
            print(
                f"  {key:<9} n={stats.count:<4} mean {stats.mean_return:+.3f}%  "
                f"median {stats.median_return:+.3f}%  hit {hit}{err}"
            )
        edge = report.edge_over_base_rate()
        if edge is not None:
            sig = report.edge_is_significant()
            verdict = "significant at 2 se" if sig else "NOT significant at 2 se"
            print(f"  bullish edge over base rate: {edge:+.1f} pts  ({verdict})")
        share = report.no_view_share
        if share is not None:
            print(f"  no directional read: {share:.1%} of sessions")
        print(
            f"  tradeable signals (chain present): {report.tradeable_views}"
            "  — zero is expected; NSE serves no historical chains"
        )
        print()

    final = evaluate(cycles, horizon=1, all_bars=bars)
    n = final.smallest_directional_sample
    if n < 30:
        print(
            f"!! smallest directional sample is {n}. Nothing below ~30 either "
            "side supports a conclusion; read the shape, not the numbers.\n"
        )
    print("regimes classified:")
    for key, count in sorted(final.by_regime.items(), key=lambda kv: -kv[1]):
        print(f"  {key:<16} {count:>4}  ({count / len(cycles):.0%})")
    print("strategies selected:")
    for key, count in sorted(final.by_strategy.items(), key=lambda kv: -kv[1]):
        print(f"  {key:<16} {count:>4}  ({count / len(cycles):.0%})")


if __name__ == "__main__":
    asyncio.run(main())
