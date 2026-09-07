# Webhook to API

Some services will only ever POST at you. TradingView alerts, broker
callbacks, a CI hook, a payment notification — push, and no way to ask.

Some consumers can only ever poll. A cron job. A Google Sheet. A script on
a laptop behind NAT. A phone. A backtest reading yesterday.

The gateway sits between them:

```
POST /hook/<slug>          the sender's URL      ingest_secret
  └─▶ authenticate → strip credentials → store → cursor

GET  /v1/<slug>?since=N    the consumer's URL    read_token
GET  /v1/<slug>/latest     newest, with envelope
GET  /v1/<slug>/payload    newest payload alone, for curl | jq
GET  /v1/endpoints         what this token reads, and how stale each is
```

```bash
python -m index_option_brain.integrations.webhooks   # :8788
```

The page at `/` is the operator console: endpoints with freshness, a live
delivery feed, copyable URLs, generated client code for curl / Python /
Node / Apps Script / TradingView, and a payload inspector.

## Why it is a separate process

Same two reasons the TradingView receiver is:

- The console API is provably read-only — every route GET, HEAD or
  OPTIONS, with a test asserting it. That is the boundary the assistant
  sits behind, and the first POST there would delete the proof for every
  route at once.
- A webhook endpoint must answer the public internet. The console is
  tailnet-only precisely so it does not.

Three processes, three exposure profiles:

| Service | Bound to | Holds |
|---|---|---|
| `index-option-brain` | loopback, behind Tailscale | broker credentials, read-only API |
| `index-brain-tradingview` | public | strict chart alerts only |
| `index-brain-gateway` | public | any sender, serves deliveries back |

The gateway **service** cannot reach an order path — an import-graph test
walks `gateway.py`, `endpoints.py` and `store.py` and fails the build if
any of them imports `execution`, `risk`, a broker, an order, or the signal
relay. A delivery reaching a `raw` or `tradingview` endpoint cannot become
a trade.

That is no longer the whole story. `__main__.py` wires in the signal
relay, so a delivery to a **`strategy`** endpoint with an enabled route
can cause an outbound broker order — see
[docs/strategy-signals.md](strategy-signals.md). A second test asserts
that `__main__.py` is the *only* file in the package that imports the
relay, so "can this place an order" stays a question with one file as its
answer.

## The URL to paste into TradingView

```
https://<your-host>/hook/<slug>
```

Three things about that host, all of which bite before anything else
works.

**Ports 80 and 443 only.** TradingView's documentation says so and no
setting changes it. The gateway binds 8788, so an alert cannot reach it
directly — something must terminate TLS on 443 and forward. Right shape
anyway: the ingest secret travels in the request *body*, so a plain-HTTP
hop would put a credential on the wire.

**Webhooks need a paid TradingView plan.** Essential and above. On the
free tier the *Webhook URL* checkbox is present and does nothing.

**Behind a proxy, set `WEBHOOK_TRUST_FORWARDED_FOR=1`.** Every request's
peer address is then the proxy, so an endpoint's `allowed_ips` list of
TradingView's egress addresses matches nothing and rejects everything —
with a 401 that looks exactly like a wrong secret. The gateway logs a
warning at startup when an allowlist is set and this is not.

### The domain: `hooks.neofl.site`

Everything in this repo is pointed at that name. A **subdomain**, not the
apex, for one concrete reason: `neofl.site` already resolves to
`103.216.171.56`, which is not the trading box — so putting the webhook on
the apex would mean repointing whatever that is. `hooks.neofl.site` does
not exist yet, so it is free to take.

**Path A — Caddy on the VPS.** One DNS record, then the certificate takes
care of itself:

```
A    hooks.neofl.site    151.243.146.9
```

```bash
sudo cp deploy/Caddyfile /etc/caddy/Caddyfile   # already names the host
sudo systemctl reload caddy
python scripts/domain_check.py \
    --host hooks.neofl.site --expect-ip 151.243.146.9
```

Ports 80 **and** 443 have to be reachable. 80 is not optional even though
TradingView will use 443: Caddy needs it for the ACME HTTP-01 challenge,
so without it the certificate never issues and never renews.

**Path B — Cloudflare Tunnel.** No open ports, no A record to manage:

```bash
cloudflared tunnel login
cloudflared tunnel create index-brain
cloudflared tunnel route dns index-brain hooks.neofl.site
sudo cp deploy/cloudflared-config.yml /etc/cloudflared/config.yml
sudo cloudflared service install
```

`tunnel route dns` creates the record itself — **but only if
`neofl.site`'s nameservers are already Cloudflare's.** The apex resolving
to `103.216.171.56` suggests they are not, in which case either move the
domain's DNS to Cloudflare or take Path A and add the A record at your
current host.

Either way, set `WEBHOOK_TRUST_FORWARDED_FOR=1` — both paths put something
in front, so every request's peer address is that thing rather than
TradingView.

### Check it before touching TradingView

```bash
python scripts/domain_check.py --host hooks.neofl.site --slug tradingview
```

Six read-only checks, in the order the request travels: DNS (and whether
it is the machine you meant), port 80, port 443, the certificate,
`/health`, and `/hook/<slug>`.

The certificate check is the one that earns the script. **TradingView
refuses a self-signed or mismatched certificate outright and tells you
nothing** — the URL works perfectly in your browser once you click
through the warning, and the webhook will never arrive. It also reports
days-to-expiry, because a relay whose certificate quietly expires is a
strategy that stops trading on a Tuesday for a reason nobody looks for.

Each failure names the layer, since they look alike from TradingView's
side: DNS pointing at the old host answers 404, a closed 443 answers
nothing, a live proxy over a dead gateway answers 502, and a wrong slug
answers 404 as well.

### Getting a host

| | Use when | Cost |
|---|---|---|
| **Cloudflare Tunnel** | no public IP, no open ports, no DNS you control | free |
| **Caddy on a VPS** | you have a domain pointing at the box and 80/443 open | free + the box |

Fastest possible check that the whole chain works, no account needed:

```bash
cloudflared tunnel --url http://127.0.0.1:8788
```

It prints `https://<random-words>.trycloudflare.com`, and your webhook URL
is that plus `/hook/<slug>`. **The URL changes every restart** — a quick
tunnel is for testing, and an alert pointing at a dead hostname fails
silently from the chart's side. `deploy/cloudflared-config.yml` has the
named-tunnel version with a stable hostname; `deploy/Caddyfile` is the
VPS-with-a-domain path, using Caddy rather than nginx because it renews
the certificate itself — a relay whose certificate quietly expires is a
strategy that stops trading on a Tuesday for a reason nobody looks for.

### Checking the URL

Open it in a browser. It answers **405** with a note saying it accepts
POST only. That is the URL being *correct*: FastAPI's bare 405 is the
least helpful thing to see at the moment you are checking whether you
copied it right, so there is a handler that explains itself. A **404**
means the slug is wrong; nothing at all means the host or the port is.

## Two credentials, never one

Each endpoint has an `ingest_secret` and a `read_token`, and the registry
refuses them if they match.

The pusher and the puller are different parties with different exposure.
TradingView's secret sits in an indicator input that every viewer of a
shared chart can read; the read token sits in your terminal history. One
credential for both would mean anyone who can see your chart can also read
every delivery the endpoint ever received.

A read token grants exactly the endpoints it belongs to. There is no admin
token that reads everything: one leaked credential should cost one
endpoint, not the gateway.

## Endpoints are configured, not created from the page

The web page builds the registry entry; it does not install it. Adding
public ingress to a running system is an operator action on the machine —
the same rule the git control loop applies to `run_mode`. A page that
mints new POST endpoints on demand is a page that can be used to mint them
by whoever reaches it.

```bash
cp deploy/webhook-endpoints.example.json var/webhook-endpoints.json
chmod 600 var/webhook-endpoints.json     # the gateway refuses anything looser
```

Values support `${ENV_VAR}`, so the registry can be committed while the
secrets stay in the environment.

## Credentials in the body are stripped, never stored

TradingView cannot sign a request or set a header, so its shared secret
travels inside the JSON body. Persisting the body verbatim would write that
credential into the database and then serve it back over the read API to
anyone holding a read token — turning the weaker credential into the
stronger one.

Every field whose **whole name** looks like a credential (`secret`,
`token`, `api_key`, `signature`, `authorization`, …) is removed before
storage, recursively, through nested objects and lists. Whole name, not
substring: `secret_santa_id` is content, and dropping it would silently
lose data the consumer needs.

## The cursor is an integer

`seq` is a monotonic autoincrementing key, so `since` is an exact `>`. A
poller that stores `next_cursor` and passes it back sees every delivery
exactly once.

This is not the obvious choice — a received-at timestamp looks natural —
and it is here because the timestamp version was already tried in this
codebase. It had to be made **inclusive** (`>=`) to stop two deliveries in
the same second from losing one permanently: in the table, past the
cursor, never returned. Inclusive then re-reads the boundary row on every
poll. An integer sequence has neither problem.

`next_cursor` is echoed back even on an empty page, so a poller can store
one value unconditionally instead of branching on the empty case and
accidentally rewinding.

## What each failure looks like

| What happened | Response | Why that shape |
|---|---|---|
| delivered | 200 `{ok, seq}` | the seq is the cursor position |
| alert failed validation | 200 `{ok, seq, handler}` | the reason reaches TradingView's alert log |
| handler raised | 200 `{ok, seq, handler:"failed"}` | the delivery is already durable and readable |
| unknown slug | 404 | a slug is not a secret; identical answers cannot be debugged |
| wrong ingest secret / blocked IP | 401 `rejected` | no reason given |
| body over 64 KB | 413 | checked on the header and again after read |
| more than 120/min | 429 | bounds the work; retention bounds the disk |
| no / wrong read token | 401 | per-endpoint, constant-time |
| nothing delivered yet | 200 `{delivery: null}` | a 404 would look like a wrong URL |

## Four body shapes, and none of them refused

A JSON object is used as-is. A JSON array or scalar is wrapped as
`{"value": …}`, so a stored row is always an object and consumers need one
shape. Form encoding is parsed flat. **Anything else is kept as
`{"text": …}`** rather than rejected — a body this service cannot parse is
still a delivery that happened, and losing it would hide the sender's
misconfiguration instead of showing it.

The declared content type is a hint, not the decision: TradingView sends
JSON as `text/plain`, and routing on the header would put every alert in
the text branch.

## Retention

Each endpoint keeps its newest `retain` deliveries, enforced on every
insert rather than by a nightly job. A sender misconfigured to fire every
second fills a disk in a day, and the consequence lands on the engine
sharing that disk — a full disk stops the system from gating a trade,
which is the failure this project spends the most effort avoiding.

## One page, two homes

`static/gateway.html` is a **fragment** — no doctype, no `html` or `head`
element. The gateway wraps it in a minimal document shell when serving it,
which is what lets the same file also be published as an artifact where
the host supplies that shell. One page, no second copy to keep in step.

It probes `GET /health` on load. Answered, it runs in **live mode** —
token box, endpoint list, polling feed. Unanswered (published as an
artifact, opened from disk), it runs in **builder mode**: generate a
registry entry with real `crypto.getRandomValues` credentials, get the
URLs and the client snippets, inspect a pasted payload.

Served same-origin, so its `fetch` calls need no CORS header. Adding one
would mean any page anywhere could be made to read this gateway with a
token its viewer pasted somewhere else.

## Forwarding: built, on the terms named here

The other reading of "convert a webhook" is to relay it onward — receive,
transform, POST somewhere else. That exists now for `strategy` endpoints,
and it follows the rule this file set out before it did: **the sender picks
what, the operator picks where.** The destination URL, headers and body
template live in `var/signal-routes.json` on the machine, and there is
deliberately no way to name a destination from the payload. An inbound
endpoint that will POST to any URL its caller supplies is an open relay.

See [docs/strategy-signals.md](strategy-signals.md).
