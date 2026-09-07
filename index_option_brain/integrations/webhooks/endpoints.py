"""Which inbound endpoints exist, and the two credentials each one has.

Endpoints are declared in configuration, not created by clicking a button
in the web page. That is a deliberate restriction and it is the same rule
the git control loop follows for `run_mode`: **adding public ingress to a
running system is an operator action on the machine, not a click in a
UI.** A page that mints new unauthenticated POST endpoints on demand is a
page that can be used to mint them by anyone who reaches it.

Two credentials, never one
--------------------------
Each endpoint has an `ingest_secret` and a `read_token`, and they must
differ. The party that pushes (TradingView, a broker's callback, a cron on
someone else's box) and the party that polls (your script, the page in
your browser) are different parties with different exposure. TradingView's
secret sits in an indicator input that every viewer of a shared chart can
read; the read token sits in your terminal history. Sharing one credential
between those two would mean anyone who can see your chart can also read
every delivery the endpoint ever received.

`kind` decides what happens after authentication
------------------------------------------------
`raw` stores the payload and serves it back — the general push-to-pull
bridge. `tradingview` additionally runs the strict alert validation in
`integrations.tradingview`, so a chart alert still cannot claim a trigger
a chart could not observe. A `raw` endpoint deliberately validates
nothing: it exists to carry payloads whose shape this system does not
know, and a gateway that rejected what it did not understand would be
useless for exactly that job.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

#: A slug appears in a public URL, so it is restricted to what is safe
#: there and short enough to type. Refused rather than sanitised: quietly
#: rewriting `My Hook` to `my-hook` means the URL you were given and the
#: URL that works are different strings.
SLUG_ALLOWED = set("abcdefghijklmnopqrstuvwxyz0123456789-_")

MIN_CREDENTIAL_LENGTH = 16

#: Deliveries kept per endpoint. A sender misconfigured to fire every
#: second fills a disk in a day otherwise, and the failure would land on
#: the engine sharing that disk rather than on the gateway.
DEFAULT_RETAIN = 500


class EndpointKind(StrEnum):
    RAW = "raw"
    TRADINGVIEW = "tradingview"


@dataclass(frozen=True)
class Endpoint:
    slug: str
    ingest_secret: str
    read_token: str
    kind: EndpointKind = EndpointKind.RAW
    retain: int = DEFAULT_RETAIN
    #: Source addresses allowed to POST here. Empty means any address, which
    #: is correct behind a tunnel and for a sender that publishes no egress
    #: range — the secret is then the only credential, which is why it has a
    #: minimum length.
    allowed_ips: frozenset[str] = frozenset()
    description: str = ""

    def __post_init__(self) -> None:
        if not self.slug or not set(self.slug) <= SLUG_ALLOWED:
            raise ValueError(
                f"endpoint slug {self.slug!r} must be lower-case letters, "
                "digits, hyphen or underscore"
            )
        if len(self.ingest_secret) < MIN_CREDENTIAL_LENGTH:
            raise ValueError(
                f"{self.slug}: ingest secret must be at least "
                f"{MIN_CREDENTIAL_LENGTH} characters"
            )
        if len(self.read_token) < MIN_CREDENTIAL_LENGTH:
            raise ValueError(
                f"{self.slug}: read token must be at least "
                f"{MIN_CREDENTIAL_LENGTH} characters"
            )
        if self.ingest_secret == self.read_token:
            # The whole reason there are two. See the module docstring.
            raise ValueError(
                f"{self.slug}: the ingest secret and the read token must "
                "differ — the pusher and the poller are different parties"
            )
        if self.retain < 1:
            raise ValueError(f"{self.slug}: retain must be at least 1")


def _endpoint_from(slug: str, spec: dict[str, Any]) -> Endpoint:
    raw_ips = spec.get("allowed_ips") or []
    if isinstance(raw_ips, str):
        raw_ips = [part.strip() for part in raw_ips.split(",") if part.strip()]
    return Endpoint(
        slug=slug,
        ingest_secret=str(spec.get("ingest_secret", "")),
        read_token=str(spec.get("read_token", "")),
        kind=EndpointKind(str(spec.get("kind", "raw"))),
        retain=int(spec.get("retain", DEFAULT_RETAIN)),
        allowed_ips=frozenset(str(ip) for ip in raw_ips),
        description=str(spec.get("description", "")),
    )


def load(path: str | Path) -> dict[str, Endpoint]:
    """Read the endpoint registry from a JSON file.

    The file holds credentials, so a permissive mode is refused rather than
    warned about: a world-readable secrets file on a box that also runs an
    assistant with shell access is not a warning-level problem.

    Environment expansion is supported per value (`"${TV_SECRET}"`) so the
    registry can live in git while the secrets do not.
    """
    file = Path(path)
    if not file.exists():
        return {}
    mode = file.stat().st_mode & 0o077
    if mode:
        raise PermissionError(
            f"{file} is readable or writable beyond its owner (mode "
            f"{oct(file.stat().st_mode & 0o777)}); it holds credentials — "
            "chmod 600 it"
        )
    document = json.loads(file.read_text())
    if not isinstance(document, dict):
        # ValueError, not TypeError: nobody passed a wrong argument, the
        # file's contents are wrong — and the runner catches the config
        # family (ValueError, PermissionError, OSError) to exit 2, which is
        # exactly what a bad registry should do.
        raise ValueError(  # noqa: TRY004
            f"{file} must contain a JSON object of endpoints"
        )

    endpoints: dict[str, Endpoint] = {}
    for slug, spec in document.items():
        if not isinstance(spec, dict):
            raise ValueError(  # noqa: TRY004 - a config value, not an argument type
                f"{file}: endpoint {slug!r} must be an object"
            )
        expanded = {
            key: os.path.expandvars(value) if isinstance(value, str) else value
            for key, value in spec.items()
        }
        endpoints[slug] = _endpoint_from(slug, expanded)
    return endpoints
