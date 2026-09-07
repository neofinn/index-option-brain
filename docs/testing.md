# Testing the chain

Five links — TradingView, the proxy on 443, the gateway, the relay, the
broker or EA — and when an alert does nothing, none of them says which one
failed. This is how to bisect it.

## In this repo

### `scripts/hook_test.py` — the whole chain, one command

```bash
python scripts/hook_test.py \
    --base http://127.0.0.1:8788 \
    --ingest-secret "$INGEST" --read-token "$READ" \
    --slug tradingview --strategy-slug ea \
    --strategy-ingest-secret "$STRATEGY_INGEST"
```

22 cases: a valid indicator alert, a redelivery, a wrong secret, a claim a
chart cannot observe, an unmapped ticker, a stale alert, an unparseable
body, the read API, the strategy leg, the EA CSV feed, and an
unauthenticated read. Each failure prints why the case matters, not just
that it failed.

It sends `Content-Type: text/plain`, because that is what TradingView
sends. A harness that declared `application/json` would pass against a
gateway that only handled JSON content types — and TradingView would then
fail against the same gateway.

Point `--base` at the tunnel hostname to test the proxy hop too. A status
of `0` on the first case means the host, the port or the proxy; everything
after it is the gateway.

### `scripts/mock_broker.py` — a broker that logs and places nothing

```bash
python scripts/mock_broker.py                 # 200 to everything
python scripts/mock_broker.py --status 429    # rehearse a rate limit
python scripts/mock_broker.py --delay 8       # rehearse a timeout
python scripts/mock_broker.py --fail-every 3  # rehearse intermittency
```

Point a route's `destination.url` at `http://127.0.0.1:9099/orders` and
set `enabled: true`. Loopback http is allowed for exactly this. The relay
then runs the whole live path — renders the template, sets the headers,
POSTs — and you read the exact body a real broker would have received.

That is a different thing from the relay's own dry run. A dry run proves
the relay *would* have built the right request. This proves the request
survives the wire and is shaped the way your broker's documentation says.
The gap between those two is where "the template looked fine" lives — the
`exit_body_template` requirement was found by reading this script's output
and noticing `{"transactionType": "EXIT", "quantity": 0}`, which no broker
accepts and a mock happily answers 200 to.

Header values are redacted in its output. It is meant to be left running
in a terminal, and a terminal is a place credentials get screenshotted out
of.

### `pytest`

1647 tests. `tests/signals/` covers the relay's guards, the idempotency
constraint and the fault attribution; `tests/integrations/webhooks/`
covers the gateway. Neither replaces a live run: the freshness gap in the
gateway path and the exit defect both passed the unit suite and were found
by the harness against a running process.

## Third-party, in the order you need them

**Before your gateway exists — [webhook.site](https://webhook.site)**
Open it, copy the URL it gives you, paste that into TradingView's *Webhook
URL* field, and fire an alert. It shows the exact body, every header, and
the source IP. This answers the two questions nothing else can: **is my
plan actually sending webhooks**, and **what does TradingView really put in
the body**. Do this first. If nothing arrives here, no amount of gateway
debugging will help. RequestBin, Beeceptor and Pipedream do the same job.

**To reach localhost — `cloudflared` or `ngrok`**
`cloudflared tunnel --url http://127.0.0.1:8788` needs no account and
gives you a 443 hostname. `ngrok http 8788` gives the same plus a request
inspector at `http://127.0.0.1:4040`, which shows every inbound request
and lets you **replay** one — the fastest way to iterate on a payload
without re-firing a chart alert.

**To hand-craft requests — Bruno, Hoppscotch, Postman or HTTPie**
Bruno keeps collections as files, so a request collection can live in the
repo rather than in someone's cloud account. HTTPie is the shortest path
from a shell:

```bash
http POST :8788/hook/tradingview \
  X-Webhook-Secret:"$INGEST" \
  kind=BREAKOUT ticker=NIFTY interval=5 price=24025.65 \
  bar_time="$(date -u +%FT%TZ)" fired_at="$(date -u +%FT%TZ)"
```

**To read the API — `curl` and `jq`**
```bash
curl -s -H "Authorization: Bearer $READ" \
  "$BASE/v1/tradingview/payload" | jq .
```

**To see what the relay sends over TLS — `mitmproxy`**
Only when a real broker is rejecting something and you cannot tell why
from the audit row. `mitmproxy` in reverse mode sits between the relay and
the broker and shows the decrypted request. Remember it holds your broker
token in its log.

**TradingView's own alert log**
The Alerts panel has a log tab. It records delivery failures — and this is
why the gateway answers **422** for a rejection the sender caused rather
than 200 with a note in the body. A 200 shows as a green tick there for an
alert that did nothing.

## Bisecting a dead alert

| Symptom | Check | Usually |
|---|---|---|
| nothing in TradingView's alert log | plan tier | webhooks need a paid plan |
| log shows a failure, gateway log empty | webhook.site test | the URL, the port, or the tunnel is down |
| gateway 401, secret looks right | `WEBHOOK_TRUST_FORWARDED_FOR` | behind a proxy the peer is the proxy, so an IP allowlist rejects everything |
| gateway 422 `NOT_CHART_OBSERVABLE` | the `kind` in your template | a chart cannot claim IV or breadth |
| gateway 422 `STALE` | the clock, or a slow tunnel | TradingView does not retry |
| gateway 200, EA sees nothing | `/v1/<slug>/routes` | the route is in dry run, which feeds nothing |
| relay `BLOCKED`, reason names a ceiling | the route's guards | your own policy, working |
| broker answers 400 | `scripts/mock_broker.py` | the body template |
