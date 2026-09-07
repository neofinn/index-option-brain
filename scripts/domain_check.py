#!/usr/bin/env python3
"""Prove the webhook URL will work before pasting it into TradingView.

TradingView tells you almost nothing when a webhook fails. The alert log
says the delivery failed and stops there, so every layer below it has to
be checked separately — and each layer has a failure that looks like a
different layer's problem:

* DNS pointing at the old host, so the request reaches something else
  entirely and answers 404;
* port 443 closed, so nothing answers and the alert log says only
  "failed";
* a **self-signed or expired certificate**, which TradingView refuses
  outright. This is the one worth checking most: the URL works perfectly
  in your browser once you click through the warning, and TradingView
  never will;
* the proxy up but the gateway behind it down, which answers 502;
* the right host and a wrong slug, which answers 404.

    python scripts/domain_check.py --host hooks.neofl.site
    python scripts/domain_check.py --host hooks.neofl.site \
        --expect-ip 151.243.146.9 --slug tradingview

Standard library only, and every check is read-only.
"""

from __future__ import annotations

import argparse
import json
import socket
import ssl
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime

GREEN, RED, YELLOW, DIM, BOLD, OFF = (
    "\033[32m",
    "\033[31m",
    "\033[33m",
    "\033[2m",
    "\033[1m",
    "\033[0m",
)

results: list[tuple[str, str, str]] = []   # (level, check, detail)


def ok(check: str, detail: str) -> None:
    results.append(("ok", check, detail))


def warn(check: str, detail: str) -> None:
    results.append(("warn", check, detail))


def bad(check: str, detail: str) -> None:
    results.append(("bad", check, detail))


def check_dns(host: str, expect_ip: str | None) -> str | None:
    try:
        addresses = sorted({info[4][0] for info in socket.getaddrinfo(host, None)})
    except socket.gaierror as exc:
        bad("DNS", f"{host} does not resolve ({exc.strerror or exc}). Add an A record.")
        return None
    joined = ", ".join(addresses)
    if expect_ip and expect_ip not in addresses:
        # The failure that looks like every other failure: the name
        # resolves, so something answers, and it is not your gateway.
        bad(
            "DNS",
            f"{host} -> {joined}, but you expected {expect_ip}. "
            "Requests are reaching a different machine.",
        )
    else:
        ok("DNS", f"{host} -> {joined}")
    return addresses[0]


def check_port(host: str, port: int, *, required: bool) -> bool:
    try:
        with socket.create_connection((host, port), timeout=6):
            ok(f"port {port}", "open")
            return True
    except OSError as exc:
        message = f"cannot connect ({exc.strerror or exc})"
        if required:
            bad(f"port {port}", f"{message}. TradingView calls 443 and 80 only.")
        else:
            warn(
                f"port {port}",
                f"{message}. Caddy needs 80 for the ACME HTTP-01 challenge, "
                "so a certificate will not renew without it.",
            )
        return False


def check_certificate(host: str) -> None:
    context = ssl.create_default_context()
    try:
        with (
            socket.create_connection((host, 443), timeout=8) as raw,
            context.wrap_socket(raw, server_hostname=host) as tls,
        ):
            certificate = tls.getpeercert()
            protocol = tls.version()
    except ssl.SSLCertVerificationError as exc:
        # Deliberately the loudest failure in this script. A browser lets
        # you click past this; TradingView does not, and gives no reason.
        bad(
            "TLS",
            f"the certificate is not trusted ({exc.verify_message or exc}). "
            "TradingView refuses a self-signed or mismatched certificate "
            "outright — a browser warning you can click through is still a "
            "webhook that will never arrive.",
        )
        return
    except OSError as exc:
        bad("TLS", f"handshake failed ({exc})")
        return

    if not certificate:
        warn("TLS", "connected but the peer returned no certificate detail")
        return

    subject = dict(x[0] for x in certificate.get("subject", ()))  # type: ignore[misc]
    issuer = dict(x[0] for x in certificate.get("issuer", ()))  # type: ignore[misc]
    names = [value for key, value in certificate.get("subjectAltName", ()) if key == "DNS"]
    expires_raw = certificate.get("notAfter")
    detail = f"{protocol}, issued by {issuer.get('organizationName') or issuer.get('commonName') or '?'}"
    if expires_raw:
        expires = datetime.strptime(str(expires_raw), "%b %d %H:%M:%S %Y %Z").replace(
            tzinfo=UTC
        )
        days = (expires - datetime.now(UTC)).days
        detail += f", {days}d left"
        if days < 0:
            bad("TLS", f"the certificate expired {abs(days)} days ago")
            return
        if days < 14:
            warn(
                "TLS",
                f"{detail} — a relay whose certificate expires stops trading "
                "for a reason nobody looks for. Check renewal.",
            )
            return
    if host not in names and not any(
        n.startswith("*.") and host.endswith(n[1:]) for n in names
    ):
        bad("TLS", f"the certificate covers {names}, not {host}")
        return
    del subject
    ok("TLS", detail)


def fetch(url: str, timeout: float = 10.0) -> tuple[int, str]:
    request = urllib.request.Request(url, headers={"User-Agent": "domain-check"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read(4096).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(4096).decode("utf-8", "replace")
    except (urllib.error.URLError, OSError) as exc:
        return 0, str(exc)


def check_gateway(host: str, slug: str) -> None:
    status, body = fetch(f"https://{host}/health")
    if status == 0:
        bad("gateway", f"/health unreachable ({body})")
        return
    if status == 502 or status == 503:
        bad(
            "gateway",
            f"the proxy answered {status} — it is up but the gateway behind "
            "it is not. Check the service, and that the proxy forwards to "
            "the port the gateway binds.",
        )
        return
    if status != 200:
        warn("gateway", f"/health answered {status}, expected 200: {body[:120]}")
        return
    try:
        health = json.loads(body)
    except json.JSONDecodeError:
        warn("gateway", f"/health is not the gateway's ({body[:80]})")
        return
    if health.get("status") != "ok":
        warn("gateway", f"/health said {body[:120]}")
        return
    ok("gateway", f"{health.get('endpoints', '?')} endpoints configured")

    status, body = fetch(f"https://{host}/hook/{slug}")
    if status == 405:
        # The URL being right. A GET on a POST-only path is what a person
        # does while checking they copied it correctly.
        ok("webhook URL", f"https://{host}/hook/{slug} — paste this into TradingView")
    elif status == 404:
        bad(
            "webhook URL",
            f"no endpoint named {slug!r}. Check the slug in "
            "var/webhook-endpoints.json.",
        )
    elif status == 0:
        bad("webhook URL", f"unreachable ({body})")
    else:
        warn("webhook URL", f"answered {status}, expected 405: {body[:120]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="hooks.neofl.site")
    parser.add_argument(
        "--expect-ip",
        default=None,
        help="the machine the name should resolve to, e.g. the VPS address",
    )
    parser.add_argument("--slug", default="tradingview")
    args = parser.parse_args()

    print(f"{BOLD}Checking https://{args.host}/hook/{args.slug}{OFF}\n")

    if check_dns(args.host, args.expect_ip) is not None:
        # 80 is not required for TradingView (it will call 443), but Caddy
        # needs it to renew, so it is a warning rather than a failure.
        check_port(args.host, 80, required=False)
        if check_port(args.host, 443, required=True):
            check_certificate(args.host)
            if all(level != "bad" for level, check, _ in results if check == "TLS"):
                check_gateway(args.host, args.slug)

    width = max(len(check) for _, check, _ in results) if results else 0
    for level, check, detail in results:
        mark = {"ok": f"{GREEN}ok  {OFF}", "warn": f"{YELLOW}warn{OFF}", "bad": f"{RED}FAIL{OFF}"}[level]
        print(f"{mark}  {check.ljust(width)}  {detail}")

    failures = sum(1 for level, _, _ in results if level == "bad")
    warnings = sum(1 for level, _, _ in results if level == "warn")
    print(f"\n{BOLD}{len(results) - failures - warnings} ok, {warnings} warnings, {failures} failed{OFF}")
    if failures:
        print(f"{DIM}Fix these before touching the TradingView alert — its log "
              f"will only tell you the delivery failed.{OFF}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
