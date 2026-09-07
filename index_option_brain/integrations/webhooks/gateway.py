"""The push-to-pull gateway: a webhook goes in, an API comes out.

Plenty of services will only ever POST at you — TradingView alerts, broker
callbacks, a CI hook, a payment notification. Plenty of consumers can only
ever poll: a cron job, a spreadsheet, a script on a laptop behind NAT, a
phone. This service sits between them. It accepts the POST, authenticates
it, stores it, and serves it back over a cursor-paged REST API that anyone
who can make a GET request can read.

    POST /hook/<slug>            the sender's URL, ingest secret
    GET  /v1/<slug>?since=N      the consumer's URL, read token
    GET  /v1/<slug>/latest       the newest delivery
    GET  /v1/<slug>/payload      just the newest payload, for curl | jq
    GET  /v1/endpoints           what this token can read, and how stale

Why it is a separate process
----------------------------
Same two reasons the TradingView receiver is. The console API is provably
read-only — every route GET, HEAD or OPTIONS, with a test asserting it —
and that proof is the boundary the assistant sits behind; the first POST
there would delete it for every route at once. And a webhook endpoint must
answer the public internet while the console is deliberately tailnet-only.

It also holds no engine, no risk, no broker: an import-graph test fails if
any module in this package can reach one. A delivery here cannot become an
order. Turning a delivery into a decision is the engine's job, done by
polling this API like any other consumer.

It must be reached through a proxy on 443
-----------------------------------------
TradingView calls webhooks on **ports 80 and 443 only** — its own
documentation says so, and there is no setting that changes it. This
service binds a high port, so a TradingView alert cannot reach it
directly and must arrive through a reverse proxy or tunnel terminating
TLS on 443. That is the right shape anyway: the ingest secret travels in
the request body, so plain HTTP would put a credential on the wire.

The consequence for the IP allowlist is the reason
`trust_forwarded_for` exists. Behind a proxy every request's peer address
is the proxy, so an `allowed_ips` list of TradingView's egress addresses
matches nothing and the endpoint rejects everything. `X-Forwarded-For` is
client-supplied text, so trusting it is off by default and must be
switched on deliberately, with a proxy in front that overwrites the
header rather than appending to whatever the client sent.

Credentials in the body are stripped, never stored
--------------------------------------------------
TradingView cannot sign a request or set a header, so its shared secret
travels inside the JSON body. Persisting the body verbatim would write that
credential into the database and then serve it back over the read API to
anyone holding a read token — turning the weaker credential into the
stronger one. Every field whose name looks like a credential is removed
before the payload is stored.
"""

from __future__ import annotations

import hmac
import json
import logging
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from index_option_brain.integrations.webhooks.endpoints import Endpoint, EndpointKind
from index_option_brain.integrations.webhooks.store import DeliveryStore

logger = logging.getLogger(__name__)

MAX_BODY_BYTES = 65536

#: Ingest rate ceiling per endpoint. Retention already bounds the disk;
#: this bounds the work. A sender stuck in a retry loop should get a 429
#: rather than a database write every few milliseconds.
DEFAULT_RATE_LIMIT = 120
RATE_WINDOW_SECONDS = 60.0

#: Body field names that are credentials rather than content. Matched
#: case-insensitively against the whole key, not as a substring: a payload
#: legitimately carrying `secret_santa_id` is content, and dropping it
#: would silently lose data the consumer needs.
CREDENTIAL_FIELDS = frozenset(
    {
        "secret",
        "token",
        "password",
        "passwd",
        "api_key",
        "apikey",
        "api_secret",
        "apisecret",
        "access_token",
        "auth",
        "authorization",
        "signature",
        "sig",
        "hmac",
        "read_token",
        "ingest_secret",
    }
)


def strip_credentials(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove credential-looking fields, at every depth.

    Recursive because a sender that nests its auth (`{"auth": {"token":
    ...}}`) is common, and a top-level-only sweep would store the token one
    level down. Lists are walked too: a batch delivery is a list of objects
    and each one can carry its own credential.
    """

    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: clean(inner)
                for key, inner in value.items()
                if key.lower() not in CREDENTIAL_FIELDS
            }
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    cleaned = clean(payload)
    assert isinstance(cleaned, dict)
    return cleaned


def parse_body(raw: bytes, content_type: str | None) -> dict[str, Any]:
    """Turn whatever arrived into a JSON object.

    Four shapes, because senders differ and a gateway that only accepted
    well-formed JSON objects would be useless for the ones that do not send
    them:

    * a JSON object — used as-is;
    * a JSON array or scalar — wrapped as `{"value": ...}`, so the stored
      row is always an object and consumers need one shape;
    * form encoding — parsed into a flat object;
    * anything else — kept as `{"text": ...}` rather than refused. A body
      this service cannot parse is still a delivery that happened, and
      losing it would hide the sender's misconfiguration instead of
      showing it.

    The declared content type is a hint, not the decision. TradingView
    sends JSON as `text/plain`, and routing on the header would put every
    alert in the `text` branch.
    """
    text = raw.decode("utf-8", errors="replace")
    stripped = text.strip()
    if stripped.startswith(("{", "[")) or stripped in ("true", "false", "null"):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return {"text": text}
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}
    if content_type and "application/x-www-form-urlencoded" in content_type:
        return dict(parse_qsl(text, keep_blank_values=True))
    try:
        number = json.loads(stripped)
    except json.JSONDecodeError:
        return {"text": text}
    return {"value": number}


class RateLimiter:
    """A per-endpoint sliding window, in memory.

    In memory on purpose: it protects this process from a sender in a retry
    loop, and a restart resetting it is not a problem worth a Redis
    dependency. It is not a security control — the credentials are.
    """

    def __init__(self, limit: int = DEFAULT_RATE_LIMIT, *, clock: Callable[[], float] | None = None) -> None:
        self._limit = limit
        self._clock = clock or time.monotonic
        self._hits: dict[str, deque[float]] = {}

    def allow(self, key: str) -> bool:
        now = self._clock()
        window = self._hits.setdefault(key, deque())
        while window and now - window[0] > RATE_WINDOW_SECONDS:
            window.popleft()
        if len(window) >= self._limit:
            return False
        window.append(now)
        return True


def _bearer(request: Request) -> str | None:
    """The read token, from the header or the query string.

    The query string is supported because the page in a browser cannot set
    a header on its own first navigation — and it is second choice, because
    a token in a URL lands in history and in any proxy log along the way.
    """
    header = request.headers.get("authorization")
    if header and header.lower().startswith("bearer "):
        return header[7:].strip()
    token = request.query_params.get("token")
    return token.strip() if token else None


def _matches(supplied: str | None, expected: str) -> bool:
    return bool(supplied) and hmac.compare_digest(supplied or "", expected)


def _ui_path() -> Path:
    return Path(__file__).resolve().parent / "static" / "gateway.html"


#: The document shell the artifact host would otherwise supply. Kept
#: minimal and matching it — charset, viewport, no body margin — so the
#: page looks the same in both homes.
HTML_SHELL = (
    "<!doctype html><html><head>"
    '<meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width, initial-scale=1">'
    "<style>body{{margin:0}}img{{max-width:100%}}[hidden]{{display:none!important}}</style>"
    "</head><body>{body}</body></html>"
)


def source_address(
    request: Request, *, trust_forwarded_for: bool
) -> str | None:
    """The address to hold against an endpoint's allowlist.

    Without a declared proxy this is the peer address and nothing else,
    whatever headers claim. With one, the **leftmost** `X-Forwarded-For`
    entry is used: a proxy appends, so the leftmost value is the original
    client as the first trusted hop saw it.
    """
    peer = request.client.host if request.client else None
    if not trust_forwarded_for:
        return peer
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    return peer


def create_gateway_app(
    endpoints: dict[str, Endpoint],
    store: DeliveryStore,
    *,
    rate_limiter: RateLimiter | None = None,
    clock: Callable[[], datetime] | None = None,
    on_delivery: Callable[[Endpoint, dict[str, Any]], Any] | None = None,
    trust_forwarded_for: bool = False,
) -> FastAPI:
    """The gateway, over a registry, a store, and nothing else.

    `on_delivery` is the hook a specialised endpoint kind uses — it is how
    a `tradingview` endpoint gets the strict alert validation without this
    module knowing anything about option chains. It runs *after* the
    delivery is stored, so a handler that raises cannot lose the payload,
    and whatever string it returns is echoed to the sender as `handler` so
    a rejected alert says so in TradingView's own log rather than only in
    a server log nobody is reading.
    """

    app = FastAPI(
        title="Webhook to API gateway",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    limiter = rate_limiter or RateLimiter()
    now = clock or (lambda: datetime.now(UTC))

    def readable(request: Request) -> list[Endpoint]:
        """Endpoints the presented token may read.

        A token grants exactly the endpoints it is the read token for. There
        is no admin token that reads everything: one leaked credential
        should cost one endpoint, not the whole gateway.
        """
        supplied = _bearer(request)
        if not supplied:
            return []
        return [ep for ep in endpoints.values() if _matches(supplied, ep.read_token)]

    def authorize_read(request: Request, slug: str) -> Endpoint | JSONResponse:
        endpoint = endpoints.get(slug)
        if endpoint is None:
            # 404 rather than a uniform 401: a slug is not a secret (it is
            # in the URL you hand the sender), and a gateway that answers
            # every mistake identically is one nobody can debug.
            return JSONResponse({"error": "no such endpoint"}, status_code=404)
        if not _matches(_bearer(request), endpoint.read_token):
            return JSONResponse({"error": "read token required"}, status_code=401)
        return endpoint

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """A constant plus the endpoint count. No slugs, no counts, no
        timings: this answers the public internet, and what is configured
        here is information about what the operator is watching."""
        return {"status": "ok", "endpoints": len(endpoints)}

    @app.post("/hook/{slug}")
    async def ingest(slug: str, request: Request) -> JSONResponse:
        endpoint = endpoints.get(slug)
        if endpoint is None:
            return JSONResponse({"error": "no such endpoint"}, status_code=404)

        peer = source_address(request, trust_forwarded_for=trust_forwarded_for)
        if endpoint.allowed_ips and (peer is None or peer not in endpoint.allowed_ips):
            return JSONResponse({"error": "rejected"}, status_code=401)
        if not limiter.allow(slug):
            return JSONResponse({"error": "too many deliveries"}, status_code=429)

        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
            return JSONResponse({"error": "body too large"}, status_code=413)
        raw = await request.body()
        if len(raw) > MAX_BODY_BYTES:
            return JSONResponse({"error": "body too large"}, status_code=413)

        content_type = request.headers.get("content-type")
        # Parsed before the secret is checked, unavoidably: a sender that
        # cannot set headers has nowhere but the body to put its
        # credential. The size cap above is what keeps that cheap.
        payload = parse_body(raw, content_type)
        supplied = request.headers.get("x-webhook-secret") or payload.get("secret")
        if not _matches(
            supplied if isinstance(supplied, str) else None, endpoint.ingest_secret
        ):
            return JSONResponse({"error": "rejected"}, status_code=401)

        stored = strip_credentials(payload)
        seq = await store.record(
            endpoint=slug,
            payload=stored,
            received_at=now(),
            source_ip=peer,
            content_type=content_type,
            body_bytes=len(raw),
            retain=endpoint.retain,
        )

        if on_delivery is not None and endpoint.kind is not EndpointKind.RAW:
            try:
                result = on_delivery(endpoint, stored)
                if hasattr(result, "__await__"):
                    result = await result
                if isinstance(result, str):
                    return JSONResponse(
                        {"ok": True, "seq": seq, "handler": result}, status_code=200
                    )
            except Exception:
                # The delivery is already durable and readable over the API.
                # A specialised handler failing is worth a log and a flag in
                # the response, not a 5xx that tells the sender to give up
                # on a payload this service has in fact kept.
                logger.exception("delivery handler failed for %s", slug)
                return JSONResponse(
                    {"ok": True, "seq": seq, "handler": "failed"}, status_code=200
                )

        return JSONResponse({"ok": True, "seq": seq}, status_code=200)

    @app.get("/hook/{slug}")
    async def hook_help(slug: str) -> JSONResponse:
        """What a browser gets when someone pastes the webhook URL into it.

        FastAPI's bare 405 for a GET on a POST-only path is the least
        helpful answer available while someone is checking whether they
        copied the URL correctly — which is exactly when they open it in a
        browser. This says the URL is right and what has to call it.
        """
        if slug not in endpoints:
            return JSONResponse({"error": "no such endpoint"}, status_code=404)
        return JSONResponse(
            {
                "endpoint": slug,
                "accepts": "POST",
                "detail": (
                    "This URL is correct. It accepts POST only — paste it "
                    "into the sender's webhook field, not a browser. "
                    "TradingView calls webhooks on ports 80 and 443 only, "
                    "so this must be reached through a proxy on 443."
                ),
            },
            status_code=405,
        )

    @app.get("/v1/endpoints")
    async def list_endpoints(request: Request) -> JSONResponse:
        allowed = readable(request)
        if not allowed:
            return JSONResponse({"error": "read token required"}, status_code=401)
        out = []
        for endpoint in sorted(allowed, key=lambda e: e.slug):
            stats = await store.stats(endpoint.slug)
            age = (
                None
                if stats.last_received_at is None
                else round((now() - stats.last_received_at).total_seconds())
            )
            out.append(
                {
                    "slug": endpoint.slug,
                    "kind": str(endpoint.kind),
                    "description": endpoint.description,
                    "retain": endpoint.retain,
                    "ip_allowlist": sorted(endpoint.allowed_ips),
                    "count": stats.count,
                    "last_seq": stats.last_seq,
                    # The reading that matters. A sender that has stopped
                    # calling looks exactly like a quiet one until you can
                    # see how long it has been quiet.
                    "last_received_at": (
                        stats.last_received_at.isoformat()
                        if stats.last_received_at
                        else None
                    ),
                    "seconds_since_last": age,
                }
            )
        return JSONResponse({"endpoints": out})

    @app.get("/v1/{slug}")
    async def poll(slug: str, request: Request, since: int = 0, limit: int = 50) -> JSONResponse:
        endpoint = authorize_read(request, slug)
        if isinstance(endpoint, JSONResponse):
            return endpoint
        deliveries = await store.since(slug, cursor=since, limit=limit)
        stats = await store.stats(slug)
        return JSONResponse(
            {
                "endpoint": slug,
                "count": len(deliveries),
                # Echoed back even when nothing arrived, so a poller can
                # store one value unconditionally instead of branching on
                # an empty page and accidentally rewinding its cursor.
                "next_cursor": deliveries[-1].seq if deliveries else since,
                "last_seq": stats.last_seq,
                "deliveries": [d.as_dict() for d in deliveries],
            }
        )

    @app.get("/v1/{slug}/latest")
    async def latest(slug: str, request: Request) -> JSONResponse:
        endpoint = authorize_read(request, slug)
        if isinstance(endpoint, JSONResponse):
            return endpoint
        delivery = await store.latest(slug)
        if delivery is None:
            # 200 with an explicit empty state, not 404. "Nothing has
            # arrived yet" is a normal answer a consumer must handle, and a
            # 404 makes it indistinguishable from a wrong URL.
            return JSONResponse({"endpoint": slug, "delivery": None})
        return JSONResponse({"endpoint": slug, "delivery": delivery.as_dict()})

    @app.get("/v1/{slug}/payload")
    async def payload(slug: str, request: Request) -> JSONResponse:
        """The newest payload alone, for `curl … | jq .field`.

        Exists because the envelope is noise at a shell prompt, and a
        consumer forced to dig through one writes a jq expression that
        breaks the next time a field is added.
        """
        endpoint = authorize_read(request, slug)
        if isinstance(endpoint, JSONResponse):
            return endpoint
        delivery = await store.latest(slug)
        return JSONResponse(delivery.payload if delivery else {})

    @app.get("/", response_class=HTMLResponse)
    async def ui() -> HTMLResponse:
        """The operator page, served from this origin.

        The file on disk is a *fragment* — no doctype, no html or head
        element — and is wrapped here. That is what lets the same file be
        published as an artifact, where the host supplies the document
        shell: one page, two homes, and no second copy to keep in step.

        Served same-origin so its `fetch` calls to `/v1/...` need no CORS
        header. Adding one would mean any page anywhere could be made to
        read this gateway with a token its viewer pasted somewhere else.
        """
        page = _ui_path()
        if not page.exists():
            return HTMLResponse(f"<h1>UI not found</h1><p>Expected {page}</p>", status_code=404)
        return HTMLResponse(HTML_SHELL.format(body=page.read_text()))

    return app
