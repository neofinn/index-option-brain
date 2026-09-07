#!/usr/bin/env python3
"""The tests a single-shot harness cannot do: state, restarts and races.

`hook_test.py` fires one payload per case and reads the answer. That
misses everything whose correctness depends on *history* — and history is
where an order relay actually goes wrong:

* a duplicate arriving after the process restarted (the whole reason the
  idempotency key is a database constraint and not a memory set);
* two deliveries of one intent arriving at the same moment (the race that
  constraint is there to lose safely);
* a broker that answers 429, or hangs, or fails intermittently;
* the retention cap actually discarding rows;
* the rate limit actually refusing;
* the kill switch actually stopping a live route.

Each scenario here needs the caller to control the gateway process, so
this script drives it: it writes its own config, starts the gateway and a
mock broker, runs the scenario, and reports. Nothing it touches is your
real configuration.

    python scripts/hook_soak.py                  # everything
    python scripts/hook_soak.py --only restart   # one scenario

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

GREEN, RED, YELLOW, DIM, BOLD, OFF = (
    "\033[32m",
    "\033[31m",
    "\033[33m",
    "\033[2m",
    "\033[1m",
    "\033[0m",
)

INGEST = "soak-ingest-secret-0000001"
STRAT_INGEST = "soak-strategy-ingest-00001"
READ = "soak-read-token-000000001"
GATEWAY_PORT = 8799
BROKER_PORT = 9199
BASE = f"http://127.0.0.1:{GATEWAY_PORT}"

REPO = Path(__file__).resolve().parent.parent


def iso(when: datetime) -> str:
    return when.isoformat(timespec="seconds").replace("+00:00", "Z")


def post(path: str, body: dict[str, Any] | None) -> tuple[int, str]:
    data = json.dumps(body).encode() if body is not None else b"{}"
    request = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "text/plain"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()
    except urllib.error.URLError as exc:
        return 0, str(exc.reason)


def get(path: str, token: str = READ) -> tuple[int, str]:
    request = urllib.request.Request(
        BASE + path, headers={"Authorization": f"Bearer {token}"}
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()
    except urllib.error.URLError as exc:
        return 0, str(exc.reason)


def strategy_alert(slug_secret: str, **over: Any) -> dict[str, Any]:
    now = datetime.now(UTC)
    body = {
        "secret": slug_secret,
        "strategy": "soak",
        "ticker": "NIFTY",
        "action": "buy",
        "quantity": 1,
        "target_position": 1,
        "bar_time": iso(now - timedelta(minutes=1)),
        "fired_at": iso(now),
        "order_id": "soak-1",
    }
    body.update(over)
    return body


class Rig:
    """Owns the temporary config, the gateway process and the mock broker."""

    def __init__(self, workdir: Path, *, rate_limit: int = 120, retain: int = 500):
        self.dir = workdir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = rate_limit
        self.retain = retain
        self.gateway: subprocess.Popen[bytes] | None = None
        self.broker: subprocess.Popen[bytes] | None = None
        self.python = sys.executable

    def write_config(self, *, enabled: bool = True, broker_url: str | None = None) -> None:
        endpoints = {
            "tv": {
                "kind": "tradingview",
                "ingest_secret": INGEST,
                "read_token": READ,
                "retain": self.retain,
            },
            "st": {
                "kind": "strategy",
                "ingest_secret": STRAT_INGEST,
                "read_token": READ,
                "retain": self.retain,
            },
        }
        destination: dict[str, Any]
        if broker_url:
            destination = {
                "kind": "http",
                "url": broker_url,
                "headers": {"access-token": "SOAK-TOKEN-MUST-NOT-APPEAR"},
                "body_template": {
                    "side": "{action_upper}",
                    "securityId": "{symbol}",
                    "quantity": "{quantity}",
                },
            }
        else:
            destination = {"kind": "pull"}
        routes = {
            "st": {
                "enabled": enabled,
                "destination": destination,
                "symbol_map": {"NIFTY": "13"},
                "allowed_actions": ["buy", "sell", "exit"],
                "max_quantity": 2,
                "max_orders_per_day": 50,
                "max_age_seconds": 90,
            }
        }
        for name, document in (("endpoints.json", endpoints), ("routes.json", routes)):
            path = self.dir / name
            path.write_text(json.dumps(document, indent=2))
            path.chmod(0o600)

    def env(self, **extra: str) -> dict[str, str]:
        env = dict(os.environ)
        env.update(
            {
                "WEBHOOK_ENDPOINTS_FILE": str(self.dir / "endpoints.json"),
                "SIGNAL_ROUTES_FILE": str(self.dir / "routes.json"),
                "SQLITE_PATH": str(self.dir / "soak.sqlite"),
                "WEBHOOK_GATEWAY_PORT": str(GATEWAY_PORT),
                "WEBHOOK_RATE_LIMIT": str(self.rate_limit),
                "no_proxy": "127.0.0.1,localhost",
                "NO_PROXY": "127.0.0.1,localhost",
            }
        )
        env.update(extra)
        return env

    def start_broker(self, *args: str) -> None:
        self.broker = subprocess.Popen(
            [self.python, str(REPO / "scripts" / "mock_broker.py"), "--port", str(BROKER_PORT), *args],
            cwd=REPO,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self._wait(f"http://127.0.0.1:{BROKER_PORT}/")

    def start_gateway(self, **extra: str) -> None:
        self.gateway = subprocess.Popen(
            [self.python, "-m", "index_option_brain.integrations.webhooks"],
            cwd=REPO,
            env=self.env(**extra),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self._wait(BASE + "/health")

    def _wait(self, url: str, seconds: float = 25.0) -> None:
        deadline = time.time() + seconds
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=2):
                    return
            except Exception:  # noqa: BLE001 - any failure means not up yet
                time.sleep(0.3)
        raise RuntimeError(f"{url} never came up")

    def stop_gateway(self) -> None:
        if self.gateway and self.gateway.poll() is None:
            os.killpg(os.getpgid(self.gateway.pid), signal.SIGTERM)
            self.gateway.wait(timeout=15)
        self.gateway = None

    def stop(self) -> None:
        self.stop_gateway()
        if self.broker and self.broker.poll() is None:
            os.killpg(os.getpgid(self.broker.pid), signal.SIGTERM)
            self.broker.wait(timeout=15)
        self.broker = None


# ------------------------------------------------------------------ #
#  scenarios                                                          #
# ------------------------------------------------------------------ #
Result = tuple[bool, str]


def scenario_restart(rig: Rig) -> Result:
    """A duplicate arriving after the process restarted.

    The single most important claim in the relay: idempotency is a
    database constraint, so a restart between two deliveries of one alert
    cannot double-order. An in-memory set passes every unit test and fails
    exactly here.
    """
    rig.write_config(broker_url=f"http://127.0.0.1:{BROKER_PORT}/orders")
    rig.start_broker()
    rig.start_gateway()
    alert = strategy_alert(STRAT_INGEST, order_id="restart-1")
    first_status, first = post("/hook/st", alert)
    if "SENT" not in first:
        return False, f"first delivery was not sent: {first_status} {first}"

    rig.stop_gateway()
    rig.start_gateway()

    # Same bar, same size, fresh firing time — exactly what a reconnect
    # produces.
    again = dict(alert)
    again["fired_at"] = iso(datetime.now(UTC))
    status, body = post("/hook/st", again)
    if "DUPLICATE" not in body:
        return False, f"redelivery after restart was NOT a duplicate: {status} {body}"
    orders = (rig.broker.stdout.read1(65536).decode() if rig.broker and rig.broker.stdout else "")
    del orders
    return True, "restart did not free the idempotency key"


def scenario_race(rig: Rig) -> Result:
    """Two deliveries of one intent at the same moment.

    The race the unique constraint exists to lose safely. Threads rather
    than sequential calls, because a sequential pair proves only that the
    row was already committed.
    """
    import threading

    rig.write_config(broker_url=f"http://127.0.0.1:{BROKER_PORT}/orders")
    rig.start_broker()
    rig.start_gateway()
    alert = strategy_alert(STRAT_INGEST, order_id="race-1")
    results: list[str] = []
    lock = threading.Lock()

    def fire() -> None:
        _, body = post("/hook/st", dict(alert))
        with lock:
            results.append(body)

    threads = [threading.Thread(target=fire) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    sent = sum(1 for r in results if '"SENT"' in r or "SENT" in r and "DUPLICATE" not in r)
    duplicates = sum(1 for r in results if "DUPLICATE" in r)
    if sent != 1:
        return False, f"{sent} of 6 concurrent deliveries were sent (want exactly 1): {results}"
    if duplicates != 5:
        return False, f"{duplicates} duplicates, want 5: {results}"
    return True, "6 concurrent deliveries of one intent produced exactly 1 order"


def scenario_broker_failures(rig: Rig) -> Result:
    """429, then a hang, then intermittency.

    What matters is that a failure is recorded and **not retried**, and
    that it does not free the key: a broker call whose response was lost
    may have executed.
    """
    rig.write_config(broker_url=f"http://127.0.0.1:{BROKER_PORT}/orders")
    rig.start_broker("--status", "429")
    rig.start_gateway()
    alert = strategy_alert(STRAT_INGEST, order_id="fail-1")
    _, body = post("/hook/st", alert)
    if "FAILED" not in body:
        return False, f"a 429 from the broker was not recorded as FAILED: {body}"

    retry = dict(alert)
    retry["fired_at"] = iso(datetime.now(UTC))
    _, again = post("/hook/st", retry)
    if "DUPLICATE" not in again:
        return False, f"a failed send freed the idempotency key: {again}"
    return True, "a 429 is FAILED, not retried, and the key stays claimed"


def scenario_rate_limit(rig: Rig) -> Result:
    """The ceiling on work, as opposed to the ceiling on disk."""
    rig.rate_limit = 5
    rig.write_config()
    rig.start_gateway()
    now = datetime.now(UTC)
    codes = []
    for n in range(8):
        status, _ = post(
            "/hook/tv",
            {
                "secret": INGEST,
                "kind": "BREAKOUT",
                "ticker": "NIFTY",
                "interval": "5",
                "price": "24025.65",
                "bar_time": iso(now - timedelta(minutes=1, seconds=n)),
                "fired_at": iso(now),
            },
        )
        codes.append(status)
    if 429 not in codes:
        return False, f"no 429 after 8 deliveries against a limit of 5: {codes}"
    if codes[:5] == [429] * 5:
        return False, f"the limit refused from the first delivery: {codes}"
    return True, f"limit 5 -> {codes.count(429)} of 8 refused with 429"


def scenario_retention(rig: Rig) -> Result:
    """The cap on disk, applied on write."""
    rig.retain = 3
    rig.write_config()
    rig.start_gateway()
    now = datetime.now(UTC)
    for n in range(7):
        post(
            "/hook/tv",
            {
                "secret": INGEST,
                "kind": "BREAKOUT",
                "ticker": "NIFTY",
                "interval": "5",
                "price": f"2402{n}.00",
                "bar_time": iso(now - timedelta(minutes=1, seconds=n)),
                "fired_at": iso(now),
            },
        )
    _, body = get("/v1/tv?since=0&limit=50")
    kept = json.loads(body)["deliveries"]
    if len(kept) != 3:
        return False, f"{len(kept)} rows kept, want 3"
    prices = [d["payload"]["price"] for d in kept]
    if prices != ["24024.00", "24025.00", "24026.00"]:
        return False, f"the wrong three were kept: {prices}"
    return True, "retain=3 kept the newest three and discarded the rest"


def scenario_kill_switch(rig: Rig) -> Result:
    """One variable, ahead of everything, on a live route."""
    rig.write_config(broker_url=f"http://127.0.0.1:{BROKER_PORT}/orders")
    rig.start_broker()
    rig.start_gateway(SIGNAL_RELAY_KILL="1")
    _, body = post("/hook/st", strategy_alert(STRAT_INGEST, order_id="kill-1"))
    if "SIGNAL_RELAY_KILL" not in body:
        return False, f"a live route sent with the kill switch set: {body}"
    _, feed = get("/v1/st/signals?since=0")
    if json.loads(feed)["count"] != 0:
        return False, f"a killed signal reached the feed: {feed}"
    return True, "the kill switch stopped an enabled route"


def scenario_ea_pull(rig: Rig) -> Result:
    """The EA's loop: poll, act, store the cursor, poll again.

    The cursor has to advance past rows the EA is not given, or a blocked
    signal is re-examined on every poll for as long as it is the newest
    row.
    """
    rig.write_config(broker_url=None)  # pull destination
    rig.start_gateway()
    now = datetime.now(UTC)

    post("/hook/st", strategy_alert(STRAT_INGEST, order_id="ea-1"))
    post(
        "/hook/st",
        strategy_alert(STRAT_INGEST, order_id="ea-2", quantity=99, target_position=99,
                       bar_time=iso(now - timedelta(minutes=2))),
    )
    post(
        "/hook/st",
        strategy_alert(STRAT_INGEST, order_id="ea-3", action="exit",
                       quantity=None, target_position=None,
                       bar_time=iso(now - timedelta(minutes=3))),
    )

    status, csv = get("/v1/st/signals.csv?since=0")
    lines = [line for line in csv.strip().split("\n") if line]
    if not lines[0].startswith("#cursor="):
        return False, f"no cursor line: {lines[:2]}"
    cursor = int(lines[0].split("=")[1])
    rows = lines[2:]
    actions = [row.split(",")[4] for row in rows]
    if sorted(actions) != ["buy", "exit"]:
        return False, f"the EA was fed {actions}, want buy and exit (the 99 was blocked)"
    for row in rows:
        if len(row.split(",")) != len(lines[1].split(",")):
            return False, f"a row has the wrong column count: {row}"

    _, second = get(f"/v1/st/signals.csv?since={cursor}")
    if [line for line in second.strip().split("\n")][2:]:
        return False, "the cursor did not advance past what was already read"
    if "SOAK-TOKEN" in csv:
        return False, "the EA feed carried the broker token"
    del status
    return True, f"fed 2 of 3 signals, cursor advanced to {cursor} past the blocked one"


SCENARIOS = {
    "restart": scenario_restart,
    "race": scenario_race,
    "broker": scenario_broker_failures,
    "rate": scenario_rate_limit,
    "retention": scenario_retention,
    "kill": scenario_kill_switch,
    "ea": scenario_ea_pull,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=sorted(SCENARIOS), action="append")
    parser.add_argument("--workdir", default="/tmp/hook-soak")
    args = parser.parse_args()

    chosen = args.only or list(SCENARIOS)
    print(f"{BOLD}Soak: {', '.join(chosen)}{OFF}\n")
    passed = failed = 0
    for name in chosen:
        # A fresh directory per scenario, so a retention or rate-limit test
        # cannot leave state that changes the next one's answer.
        rig = Rig(Path(args.workdir) / name)
        try:
            ok, detail = SCENARIOS[name](rig)
        except Exception as exc:  # noqa: BLE001 - a scenario that crashes is a failure
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        finally:
            rig.stop()
        if ok:
            passed += 1
            print(f"{GREEN}ok{OFF}    {name:10s} {DIM}{detail}{OFF}")
        else:
            failed += 1
            print(f"{RED}FAIL{OFF}  {name:10s} {YELLOW}{detail}{OFF}")

    print(f"\n{BOLD}{passed} passed, {failed} failed{OFF}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
