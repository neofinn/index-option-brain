# Strategy signals: chart to broker, or chart to EA

A TradingView **indicator** alert is an observation — "the range broke".
A TradingView **strategy** alert is an order intent — "go long 2
contracts". They arrive through the same webhook and they are not the same
kind of thing, so they take different paths.

```
strategy alert ─▶ POST /hook/<slug>          kind: "strategy"
                    │  authenticate, strip credentials, store
                    ▼
                  signals.relay              route from var/signal-routes.json
                    │
        ┌───────────┴────────────┐
        ▼                        ▼
  destination: http        destination: pull
  POST to a broker         held for an EA to poll
                                 │
                                 ▼
                     GET /v1/<slug>/signals.csv?since=N
```

## Read this before enabling a route

**Nothing in this path consults the nine brains, the Regime Engine, the
Risk Engine or the Execution Gate.** A signal that arrives is a signal
that goes, inside the route's ceilings. Enabling a route delegates the
decision to the TradingView strategy and to nothing else in this
repository.

That is a legitimate thing to want — it is the whole point of a signal
relay — and it is also the single largest reduction in safety this
codebase has. The engine's own path is unchanged and still exists: an
alert on a `tradingview` endpoint becomes an `Event`, wakes the pipeline,
and whatever comes out still has to pass risk and the gate. A route here
bypasses all of it.

Every route is `enabled: false` until an operator changes it in a file on
the machine. `SIGNAL_RELAY_KILL=1` in the environment stops every route
regardless — one variable, one SSH command, because the moment you need it
you do not want to be editing JSON.

## Relay the target, not the action

TradingView gives you both. `{{strategy.order.action}}` is a **delta** —
buy 2, sell 1. `{{strategy.position_size}}` is the **position the strategy
believes it now holds** after that order.

Relaying deltas is the obvious design and the wrong one. Webhooks are
at-most-once, so a single lost alert leaves a delta relay permanently out
of step with the chart — and it stays wrong for every trade after it. A
relay that reconciles a target self-heals: the next signal restates the
whole truth, and the difference between the target and the live position
is the order to send.

So send `target_position` in your alert template. The relay prefers it,
the EA reconciles against it, and `action` is the fallback for a template
that does not carry it.

An `exit` needs no size at all: its target is flat, which is a fact
rather than a guess.

## Nothing is invented

A signal with neither a quantity nor a target is **refused**. Not
defaulted to one lot — a default size is a real position nobody chose,
which makes it the most expensive default in the system.

## The alert template

```json
{
  "secret": "<ingest_secret>",
  "strategy": "orb-nifty",
  "ticker": "{{ticker}}",
  "action": "{{strategy.order.action}}",
  "quantity": "{{strategy.order.contracts}}",
  "target_position": "{{strategy.position_size}}",
  "market_position": "{{strategy.market_position}}",
  "order_id": "{{strategy.order.id}}",
  "price": "{{close}}",
  "bar_time": "{{time}}",
  "fired_at": "{{timenow}}",
  "comment": "{{strategy.order.comment}}"
}
```

Field names follow TradingView's own placeholders rather than something
tidier, because whoever fills this in is reading TradingView's
documentation and should not need a translation table.

`strategy` is not decoration: the daily order cap and the audit trail are
keyed on it.

## The five ways a signal stops

| | What | Recorded as |
|---|---|---|
| 1 | `SIGNAL_RELAY_KILL` is set — checked before parsing | `BLOCKED` |
| 2 | route not `enabled` — full rehearsal, nothing sent | `DRY_RUN` |
| 3 | guard: action, symbol, quantity ceiling, staleness, daily cap | `BLOCKED` |
| 4 | duplicate intent | `DUPLICATE` |
| 5 | destination refused or unreachable | `FAILED` |

Refusals are written to `signal_dispatches` too. An audit trail that
records only what was sent cannot answer the question actually asked after
a bad day, which is what was refused and why.

A dry run is a **full rehearsal**: it validates, maps the symbol, applies
every guard, renders exactly what it would have transmitted and stores it.
A dry-run mode that exercises nothing is a mode that first runs on the day
it matters.

## Idempotency is in the database, not in memory

TradingView re-fires an alert on a reconnect and replays one on a chart
reload — both with a fresh `{{timenow}}`. The idempotency key is a hash of
the strategy, the bar, the order id and the size, and there is a unique
constraint on `(route, idempotency_key)`.

In memory this check would not survive the restart that happens between
two deliveries of one alert. In the database it is atomic, survives a
restart, and holds with two relay processes.

A `FAILED` send does **not** free the key. That is deliberate: the
previous attempt may have reached the broker even though its response was
lost, and a blind retry is how one intent becomes two positions. Retrying
is an operator decision.

## Destinations

### `http` — a broker's REST API

```json
"destination": {
  "kind": "http",
  "url": "https://api.dhan.co/v2/orders",
  "headers": { "access-token": "${DHAN_ACCESS_TOKEN}" },
  "body_template": {
    "transactionType": "{action_upper}",
    "securityId": "{symbol}",
    "quantity": "{quantity}",
    "orderType": "MARKET"
  }
}
```

Template values are rendered from a **fixed set of names** — no
expressions, no attribute access, no eval. A template language here would
make this file compute things, and the reason it is configuration is that
it does not.

Available: `strategy`, `ticker`, `symbol`, `action`, `action_upper`,
`side_bs` (B/S/X), `side_long_short`, `quantity`, `target`, `price`,
`order_id`, `comment`, `bar_time`, `fired_at`.

A value that is **exactly** one placeholder over a numeric field becomes a
JSON number; anything else is string-formatted. So `"{quantity}"` sends
`2` and `"lots={quantity}"` sends `"lots=2"` — several broker APIs reject
`"2"` where they want `2`, and an integral Decimal serialises as `2` not
`2.0`.

Plain `http://` to a remote host is refused: the headers carry a broker
token, and a relay is exactly the component nobody looks at again after it
starts working. Loopback http is allowed, for a local bridge.

Only header **names** reach the audit row. Knowing an `access-token`
header was set is what you need when a broker answers 401; the value is
the one thing that must never be stored.

### `pull` — an MT4/MT5 Expert Advisor

MetaTrader has no listening socket. It cannot be sent a webhook, so the
relay holds the signal and the EA reads it with `WebRequest` on a timer.

```
GET /v1/<slug>/signals.csv?since=N     Authorization: Bearer <read_token>

#cursor=124
seq,fired_at,strategy,symbol,action,quantity,target,price,order_id
124,2026-09-07T04:15:00+00:00,orb-nifty,NIFTY.I,buy,1,1,,
```

Two formats. `/signals` is JSON; `/signals.csv` is a cursor line, a header
line, then one row per signal. MQL has no JSON parser, and a hand-rolled
one in an order path is a bug with a position attached — so the CSV exists
and `StringSplit` handles it in four lines.

The column order is fixed and documented because an EA reads it by index.
Inserting a column in the middle would silently shift every field for
every deployed EA. Commas inside free text are replaced with semicolons
rather than quoted, because `StringSplit` has no notion of quoting.

**Only `SENT` rows are fed.** A disabled route produces `DRY_RUN` rows and
the EA sees nothing — which is the whole meaning of a dry run. Blocked and
failed rows are withheld for the same reason. The cursor still advances
past them, or a blocked signal would be re-examined on every poll for as
long as it is the newest row.

`index_option_brain/signals/mql/IndexBrainSignals.mq5` is a working
poller. It reconciles the target against the live net position and trades
the difference. `EnableTrading` is **false** by default and every decision
is printed instead.

Three things about it worth knowing before you run it:

- The gateway's base URL must be added to Tools → Options → Expert
  Advisors → *Allow WebRequest for listed URL*, or `WebRequest` returns
  −1 with error 4014. There is no way around that from inside an EA.
- It assumes a **netting** account: `PositionSelect` returns one position
  per symbol. On a hedging account the arithmetic is wrong.
- **It has not been run against a broker.** Run it on demo, read the log,
  and only then consider `EnableTrading=true`. An EA that places orders is
  not something to trust because it compiled.

## Setting it up

The webhook URL is `https://<your-host>/hook/<slug>`. TradingView calls
**ports 80 and 443 only** and needs a **paid plan**, so the gateway's own
8788 is never the port being called — see
[docs/webhook-to-api.md](webhook-to-api.md#the-url-to-paste-into-tradingview)
for the Cloudflare Tunnel and Caddy paths, and remember
`WEBHOOK_TRUST_FORWARDED_FOR=1` once something is in front.

```bash
cp deploy/webhook-endpoints.example.json var/webhook-endpoints.json
cp deploy/signal-routes.example.json     var/signal-routes.json
chmod 600 var/webhook-endpoints.json var/signal-routes.json
```

An endpoint with `kind: "strategy"` needs a route with the **same slug**.
The gateway refuses to start otherwise — a strategy endpoint with no route
accepts orders and silently discards them, which is worse than a refusal
because it answers 200.

The startup log states, for every route, whether it is live or rehearsing.
Said at every start and for both states, because a relay that is live and
one that is rehearsing are indistinguishable from the outside.

```
route ea -> pull: dry run (max qty 2, 20/day, symbols BANKNIFTY,NIFTY)
route broker -> http: LIVE — deliveries will be sent (max qty 1, 6/day, symbols NIFTY)
```

## The guards are ceilings, not a strategy

`max_quantity`, `allowed_actions`, `symbol_map` and `max_orders_per_day`
bound a strategy that has gone wrong: a loop firing every bar, a template
with a stray zero, a chart someone switched to the wrong symbol. They are
**not** position sizing. This relay does not size, and must not be read as
doing so — it forwards a size the strategy chose, inside limits you set.

`symbol_map` is exact and an unmapped ticker is refused rather than passed
through. A chart switched to the wrong symbol would otherwise send a real
order for an instrument nobody meant.

Blocked signals do not consume the daily cap. The cap bounds orders, not
attempts — a misconfigured template must not exhaust the day's allowance
without one order reaching anyone.
