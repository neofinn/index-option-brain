#!/usr/bin/env python3
"""A broker that logs what you send it and places nothing.

Point a `strategy` route's destination at this and enable the route. The
relay then does the whole live path — renders the template, sets the
headers, POSTs — and you read the exact body a real broker would have
received, before one ever does.

That is a different thing from the relay's own dry run. A dry run proves
the relay *would* have built the right request; this proves the request
survives the wire, arrives with the headers you meant, and is shaped the
way the broker's documentation says. The gap between those two is where
"the template looked fine" lives.

    python scripts/mock_broker.py                    # 200 to everything
    python scripts/mock_broker.py --status 429       # rehearse a rate limit
    python scripts/mock_broker.py --delay 8          # rehearse a timeout
    python scripts/mock_broker.py --fail-every 3     # rehearse intermittency

Standard library only, so it runs on a box with nothing installed.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REDACT = {
    "access-token",
    "authorization",
    "x-api-key",
    "api-key",
    "x-webhook-secret",
}


class Handler(BaseHTTPRequestHandler):
    status_code = 200
    delay_seconds = 0.0
    fail_every = 0
    counter = 0

    def log_message(self, fmt: str, *args: object) -> None:
        return  # the request dump below is the log

    def do_POST(self) -> None:
        Handler.counter += 1
        length = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(length) if length else b""

        stamp = datetime.now(UTC).strftime("%H:%M:%S")
        print(f"\n\033[1m#{Handler.counter}  {stamp}  POST {self.path}\033[0m")
        for name, value in self.headers.items():
            # Header values are redacted rather than printed. This script
            # exists to be left running in a terminal, and a terminal is a
            # place credentials get screenshotted out of.
            shown = "<redacted>" if name.lower() in REDACT else value
            print(f"  {name}: {shown}")
        if raw:
            try:
                print("  body:", json.dumps(json.loads(raw), indent=2)[:2000])
            except json.JSONDecodeError:
                print("  body (not JSON):", raw[:500])

        if Handler.delay_seconds:
            # A slow broker is the failure mode nobody rehearses. The relay
            # does not retry, so a timeout here shows you exactly what an
            # unanswered order looks like in the audit trail.
            print(f"  ... holding the response for {Handler.delay_seconds}s")
            time.sleep(Handler.delay_seconds)

        code = Handler.status_code
        if Handler.fail_every and Handler.counter % Handler.fail_every == 0:
            code = 502
        body = json.dumps(
            {
                "mock": True,
                "orderId": f"MOCK-{Handler.counter:04d}",
                "status": "accepted" if 200 <= code < 300 else "rejected",
            }
        ).encode()
        print(f"  -> {code}")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        body = b'{"mock":true,"hint":"POST here; this is a mock broker"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=9099)
    parser.add_argument(
        "--status", type=int, default=200, help="status to answer with"
    )
    parser.add_argument(
        "--delay", type=float, default=0.0, help="seconds to hold the response"
    )
    parser.add_argument(
        "--fail-every", type=int, default=0, help="answer 502 every Nth request"
    )
    args = parser.parse_args()

    Handler.status_code = args.status
    Handler.delay_seconds = args.delay
    Handler.fail_every = args.fail_every

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(
        f"mock broker on http://127.0.0.1:{args.port}  "
        f"(status {args.status}"
        + (f", delay {args.delay}s" if args.delay else "")
        + (f", 502 every {args.fail_every}" if args.fail_every else "")
        + ")\n"
        "Point a route's destination.url at it. Loopback http is allowed "
        "for exactly this."
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
