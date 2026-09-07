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

The gateway cannot reach an order path — an import-graph test walks the
package and fails the build if any module there imports `execution`,
`risk`, a broker or an order. A delivery cannot become a trade here.
Turning one into a decision is the engine's job, done by polling this API
like any other consumer.

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

## Deliberately not built: forwarding

The other reading of "convert a webhook" is to relay it onward — receive,
transform, POST somewhere else. Not here, because an inbound endpoint that
will POST to any URL you name is an open relay: whoever holds the ingest
secret picks the destination, and the destination list is exactly the kind
of thing that ends up edited from a web page. If it is built, the
destination allowlist belongs in the registry file beside the credentials,
and the same rule applies — configured on the machine, not from the page.
