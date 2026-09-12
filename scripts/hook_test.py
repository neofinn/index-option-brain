#!/usr/bin/env python3
"""Fire realistic TradingView payloads at a running gateway and check what
comes back.

The chain has five links — TradingView, the proxy on 443, the gateway, the
relay, the broker or EA — and when an alert does nothing there is no
message saying which link failed. This exercises every link you can reach
without TradingView itself, and names the one that broke.

    # against a local gateway
    python scripts/hook_test.py \
        --base http://127.0.0.1:8788 \
        --ingest-secret "$INGEST" --read-token "$READ"

    # against the real thing, through the tunnel
    python scripts/hook_test.py --base https://hooks.example.com ...

Every scenario is a payload a chart or a person actually produces,
including the malformed ones: an alert with a stray field, a redelivery,
a stale one, a claim a chart could not have observed. A harness that only
sends valid input tests the half of the code that was never going to be
the problem.

Standard library only, so it runs on the box.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

GREEN, RED, YELLOW, DIM, BOLD, OFF = (
    "\033[32m",
    "\033[31m",
    "\033[33m",
    "\033[2m",
    "\033[1m",
    "\033[0m",
)


def iso(when: datetime) -> str:
    return when.isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass
class Case:
    name: str
    path: str
    body: dict[str, Any] | None = None
    method: str = "POST"
    expect_status: int | tuple[int, ...] = 200
    #: Substrings that must appear in the response body. Checked rather
    #: than a full match because the gateway adds fields over time and a
    #: harness that breaks on every addition gets deleted.
    expect_in: tuple[str, ...] = ()
    expect_not_in: tuple[str, ...] = ()
    why: str = ""
    headers: dict[str, str] = field(default_factory=dict)


def call(
    base: str, case: Case, token: str
) -> tuple[int, str]:
    url = base.rstrip("/") + case.path
    data = json.dumps(case.body).encode() if case.body is not None else None
    headers = dict(case.headers)
    if case.path.startswith("/v1/") and "Authorization" not in headers:
        # setdefault, not assignment: a case that deliberately sends a bad
        # token was having it overwritten with the good one, so the
        # "unauthorized read is refused" check passed against a valid
        # request and proved nothing.
        headers["Authorization"] = f"Bearer {token}"
    if data is not None:
        # Deliberately text/plain: that is what TradingView sends, and a
        # gateway that only worked with application/json would pass a
        # harness that lied about the content type.
        headers.setdefault("Content-Type", "text/plain")
    request = urllib.request.Request(url, data=data, headers=headers, method=case.method)
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")
    except urllib.error.URLError as exc:
        return 0, f"unreachable: {exc.reason}"


def build_cases(
    *,
    ingest: str,
    hook_slug: str,
    strategy_slug: str,
    strategy_ingest: str,
) -> list[Case]:
    now = datetime.now(UTC)
    bar = iso(now - timedelta(minutes=1))
    fired = iso(now)
    # Unique per run, so a rerun is not read as a redelivery of the last
    # one — which would make every duplicate test pass for the wrong
    # reason.
    stamp = now.strftime("%H%M%S")

    def indicator(**over: Any) -> dict[str, Any]:
        body = {
            "secret": ingest,
            "kind": "BREAKOUT",
            "ticker": "NIFTY",
            "interval": "5",
            "price": "24025.65",
            "bar_time": bar,
            "fired_at": fired,
            "note": f"harness {stamp}",
        }
        body.update(over)
        return body

    def strategy(**over: Any) -> dict[str, Any]:
        body = {
            "secret": strategy_ingest,
            "strategy": f"harness-{stamp}",
            "ticker": "NIFTY",
            "action": "buy",
            "quantity": 1,
            "target_position": 1,
            "market_position": "long",
            "order_id": f"h-{stamp}-1",
            "price": "24025.65",
            "bar_time": bar,
            "fired_at": fired,
        }
        body.update(over)
        return body

    return [
        Case(
            "the URL is reachable at all",
            "/health",
            method="GET",
            expect_in=('"status":"ok"',),
            why="a 0 here means the host, the port or the proxy — not the gateway",
        ),
        Case(
            "a browser on the hook URL explains itself",
            f"/hook/{hook_slug}",
            method="GET",
            expect_status=405,
            expect_in=("POST",),
            why="405 means the URL is right; 404 means the slug is wrong",
        ),
        Case(
            "an indicator alert is accepted",
            f"/hook/{hook_slug}",
            indicator(),
            expect_in=('"handler":"accepted"',),
            why="the gateway answers {ok,seq,handler}; the standalone "
            "receiver answers {ok,accepted} — different services",
        ),
        Case(
            "the same bar redelivered is not queued twice",
            f"/hook/{hook_slug}",
            indicator(fired_at=iso(now + timedelta(seconds=2))),
            expect_in=("DUPLICATE",),
            why="TradingView re-fires on a reconnect; this must not double up",
        ),
        Case(
            "a wrong secret gets 401 with no reason",
            f"/hook/{hook_slug}",
            indicator(secret="wrong-but-long-enough"),
            expect_status=401,
            expect_not_in=("SECRET", "secret"),
            why="naming the reason tells a prober when they guessed right",
        ),
        Case(
            "a claim a chart cannot observe is refused",
            f"/hook/{hook_slug}",
            indicator(kind="IV_EXPANSION_COLLAPSE", bar_time=iso(now - timedelta(minutes=2))),
            expect_status=422,
            expect_in=("NOT_CHART_OBSERVABLE",),
            why="Pine sees one symbol's OHLCV, not the option chain",
        ),
        Case(
            "an unmapped ticker is refused, not guessed",
            f"/hook/{hook_slug}",
            indicator(ticker="NSE:FINNIFTY", bar_time=iso(now - timedelta(minutes=3))),
            expect_status=422,
            expect_in=("UNKNOWN_SYMBOL",),
        ),
        Case(
            "a stale alert is refused",
            f"/hook/{hook_slug}",
            indicator(
                # Its own bar. Sharing one with an earlier case meant the
                # dedupe check fired first and the staleness check was
                # never reached — which is how a missing freshness check
                # in the gateway path went unnoticed.
                bar_time=iso(now - timedelta(minutes=21)),
                fired_at=iso(now - timedelta(minutes=20)),
            ),
            expect_status=422,
            expect_in=("STALE",),
            why="TradingView does not retry, so a late delivery is a fault "
            "or a replay",
        ),
        Case(
            "a body that is not JSON is still kept",
            f"/hook/{hook_slug}",
            None,
            headers={"X-Webhook-Secret": ingest, "Content-Type": "text/plain"},
            expect_status=(200, 422),
            why="losing it would hide the sender's misconfiguration",
        ),
        Case(
            "the read API serves the deliveries back",
            f"/v1/{hook_slug}?since=0&limit=5",
            method="GET",
            expect_in=('"deliveries"', '"next_cursor"'),
        ),
        Case(
            "no stored payload carries the ingest secret",
            f"/v1/{hook_slug}?since=0&limit=50",
            method="GET",
            expect_not_in=(ingest,),
            why="the strongest single check in this harness",
        ),
        Case(
            "the newest payload alone, for curl | jq",
            f"/v1/{hook_slug}/payload",
            method="GET",
            expect_not_in=('"seq"',),
        ),
        # --- the strategy leg ---
        Case(
            "the relay route's state is readable",
            f"/v1/{strategy_slug}/routes",
            method="GET",
            expect_in=('"enabled"',),
            why="a dry run produces no signals, which looks like a quiet strategy",
        ),
        Case(
            "a strategy alert reaches the relay",
            f"/hook/{strategy_slug}",
            strategy(),
            expect_in=('"handler"',),
            why="SENT, DRY_RUN or a BLOCKED reason — all four are informative",
        ),
        Case(
            "the same intent again is a duplicate",
            f"/hook/{strategy_slug}",
            strategy(fired_at=iso(now + timedelta(seconds=3))),
            expect_in=("DUPLICATE",),
            why="the unique constraint, not a memory set",
        ),
        Case(
            "a size over the route ceiling is blocked",
            f"/hook/{strategy_slug}",
            strategy(quantity=999, target_position=999, order_id=f"h-{stamp}-2"),
            expect_status=200,
            expect_in=("BLOCKED",),
            why="a ceiling is the operator's policy, so 200 — reporting it "
            "as a failure would make your own guard look like a broken hook",
        ),
        Case(
            "a chart on an unmapped symbol is reported as a failure",
            f"/hook/{strategy_slug}",
            strategy(ticker="NSE:FINNIFTY", order_id=f"h-{stamp}-4"),
            expect_status=422,
            expect_in=("BLOCKED",),
            why="that one IS the sender's to fix, and invisible unless it "
            "shows red in their alert log",
        ),
        Case(
            "a signal with no size at all is refused",
            f"/hook/{strategy_slug}",
            {
                "secret": strategy_ingest,
                "strategy": f"harness-{stamp}",
                "ticker": "NIFTY",
                "action": "buy",
                "bar_time": bar,
                "fired_at": fired,
            },
            expect_status=422,
            expect_in=("BLOCKED",),
            why="a default size is a real position nobody chose",
        ),
        Case(
            "an exit is accepted without a size",
            f"/hook/{strategy_slug}",
            {
                "secret": strategy_ingest,
                "strategy": f"harness-{stamp}",
                "ticker": "NIFTY",
                "action": "exit",
                "order_id": f"h-{stamp}-3",
                "bar_time": bar,
                "fired_at": fired,
            },
            expect_status=200,
            expect_in=('"ok":true',),
            why='"close everything" is fully specified without a quantity. '
            "On a pull route this is SENT; on an HTTP route with no "
            "exit_body_template it is BLOCKED, and that is correct — an "
            "entry template renders EXIT/quantity 0, which no broker takes",
        ),
        Case(
            "the EA feed is a cursor line then CSV",
            f"/v1/{strategy_slug}/signals.csv?since=0",
            method="GET",
            expect_in=("#cursor=", "seq,fired_at"),
        ),
        Case(
            "the EA feed carries no credential",
            f"/v1/{strategy_slug}/signals.csv?since=0",
            method="GET",
            expect_not_in=(strategy_ingest,),
        ),
        Case(
            "a read without a token is refused",
            f"/v1/{hook_slug}?since=0",
            method="GET",
            headers={"Authorization": "Bearer definitely-not-the-token"},
            expect_status=401,
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="e.g. http://127.0.0.1:8788")
    parser.add_argument("--ingest-secret", required=True)
    parser.add_argument("--read-token", required=True)
    parser.add_argument("--slug", default="tradingview")
    parser.add_argument("--strategy-slug", default="ea")
    parser.add_argument(
        "--strategy-ingest-secret",
        default=None,
        help="if the strategy endpoint has its own secret",
    )
    args = parser.parse_args()

    cases = build_cases(
        ingest=args.ingest_secret,
        hook_slug=args.slug,
        strategy_slug=args.strategy_slug,
        strategy_ingest=args.strategy_ingest_secret or args.ingest_secret,
    )

    print(f"{BOLD}Testing {args.base}{OFF}\n")
    passed = failed = 0
    for case in cases:
        status, body = call(args.base, case, args.read_token)
        wanted = (
            case.expect_status
            if isinstance(case.expect_status, tuple)
            else (case.expect_status,)
        )
        problems: list[str] = []
        if status not in wanted:
            problems.append(f"status {status}, wanted {'/'.join(map(str, wanted))}")
        for needle in case.expect_in:
            if needle not in body:
                problems.append(f"missing {needle!r}")
        for needle in case.expect_not_in:
            if needle and needle in body:
                problems.append(f"LEAKED {needle!r}")

        if problems:
            failed += 1
            print(f"{RED}FAIL{OFF}  {case.name}")
            for problem in problems:
                print(f"        {problem}")
            print(f"        {DIM}{body[:300]}{OFF}")
            if case.why:
                print(f"        {YELLOW}why it matters: {case.why}{OFF}")
        else:
            passed += 1
            print(f"{GREEN}ok{OFF}    {case.name}  {DIM}{status}{OFF}")

    print(f"\n{BOLD}{passed} passed, {failed} failed{OFF}")
    if failed:
        print(
            f"{DIM}A status of 0 on the first case means the host, the port or "
            f"the proxy — TradingView calls 80 and 443 only.{OFF}"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
