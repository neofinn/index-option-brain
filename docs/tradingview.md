# TradingView on live charts

Two directions, and they are not symmetrical.

| | Direction | Mechanism | Status |
|---|---|---|---|
| **Chart → engine** | an alert fires, the engine wakes | webhook POST | built |
| **Engine → chart** | the engine's levels appear on the chart | recomputed in Pine | built, with limits |

## What can and cannot live on a chart

Pine Script sees open, high, low, close and volume for **one symbol**. That
draws a hard line through this system.

**Computable on the chart, exactly.** The whole Index brain (spec §5):
EMA separation and location, regression slope, swing structure, rate of
change, Cutler RSI, ATR and its percentile, breakout state with the live
bar excluded, session VWAP, opening range, gap. Also both realized
volatility estimators — close-to-close and Yang-Zhang. These are OHLC
arithmetic and nothing else, so the chart can carry the same number the
engine is using.

**Not computable on the chart, at all.** Everything the other three
brains read. Constituent breadth needs fifty quotes and their index
weights. Open interest, walls, max pain, OI migration, implied volatility,
the parity forward and its basis, greeks, the volatility risk premium, the
Execution Gate's sixteen checks and the cost model all need the option
chain, which TradingView does not serve for NSE options and which Pine
cannot fetch.

So the honest mapping is: **TradingView is a detector; the engine is the
decision layer.** The chart supplies price events with sub-second latency
and never misses a bar. It does not decide anything, and the indicator
says so on screen.

## Chart → engine: the webhook

### Why it is a separate process

The console API has a test asserting every route is GET, HEAD or OPTIONS.
That is the boundary the assistant sits behind — "the console cannot be
made to do anything" is a property proved by there being no verb that
could — and adding the first POST there would delete that proof for every
route at once.

Second reason: a TradingView webhook must be reachable from the public
internet, because TradingView's servers place the call. The console is
served on the tailnet precisely so that it is not. Merging them would drag
the console onto the public internet to satisfy the webhook.

So the receiver is its own ASGI app, its own process, its own port. It
holds a guard, an inbox and a sink, and nothing else — a test walks the
package's import graph and fails if any module here can reach execution,
risk, a broker or an order.

```
TradingView  ──POST──▶  receiver (:8787, public)
                            │  authenticate, validate, dedupe
                            ▼
                        system_events  ◀── the only thing crossing processes
                            │
                            ▼
                        engine + console (:8000, tailnet only)
```

### The trust model

TradingView cannot sign a webhook. No HMAC header, no mutual TLS, no
nonce — a bare POST anyone who learns the URL can reproduce. Four things
compensate, and none is sufficient alone:

1. **A shared secret in the body**, compared with `hmac.compare_digest`
   and stripped before parsing. `TradingViewAlert` has no field that could
   hold it, so the credential cannot survive into an event, a log or a
   response. Minimum sixteen characters, enforced at construction.
2. **A source-address allowlist**, defaulting to TradingView's published
   egress set. `TRADINGVIEW_TRUST_FORWARDED_FOR` is off by default:
   `X-Forwarded-For` is client-supplied text, and honouring it by default
   would let any sender name their own address.
3. **A freshness window** (120s). TradingView does not retry, so a late
   delivery is a fault or a replay.
4. **A claim check.** An alert may only assert a trigger a chart can
   observe. `IV_EXPANSION_COLLAPSE` from a chart is not a reading, it is a
   label over data the sender never had, and it is refused with
   `NOT_CHART_OBSERVABLE`.

Checks run source → secret → shape → freshness, so a disallowed address
never learns whether its guessed secret was close. A failed secret gets a
bare 401 with no reason; everything past it gets a precise diagnostic,
because that is what the operator reads out of TradingView's alert log.

### What an accepted alert becomes

An `Event`, meaning "something changed; analyze it" — the same contract
spec §4 puts on the internal trigger engine. It carries no
`significance_score`: significance is scored against this system's own
market state, and a number from the sender would be an unverified input
deciding whether the pipeline wakes.

`event_id` is a UUIDv5 over the alert's **bar** key, not its firing time,
so a redelivery across a reconnect derives the same id. That is what makes
duplicate handling work across a receiver restart: the in-memory dedupe
set is an optimisation, the deterministic id is the guarantee, and
consumers dedupe on it.

### Failure modes, and what each one looks like

| What happened | Response | Where it shows |
|---|---|---|
| accepted | 200 `accepted: true` | `/api/tradingview` on the console |
| same bar redelivered | 200 `accepted: false` | not queued twice |
| wrong secret / address | 401, no reason | guard stats |
| unobservable trigger, unknown symbol | 422 + reason | TradingView alert log |
| stale or future-dated | 409 + reason | TradingView alert log |
| database write failed | **503** | TradingView alert log |

The 503 is deliberate. The alert did reach the in-memory inbox, so a 200
would be defensible — but in a two-process deployment that inbox is a dead
end, and a green alert log while nothing downstream ever sees the alert is
the worst of the available outcomes. This is deliberately *not* the
capture recorder's fail-soft rule: capture losing a snapshot costs one row
of a corpus, a lost alert is the trigger that was supposed to wake the
analysis.

## Engine → chart: the Pine mirror

`pine/index_brain_mirror.pine` recomputes the Index brain on the chart.
It is a transliteration, not an approximation, and three of Pine's
built-ins are deliberately **not** used because they are close enough to
look right while disagreeing:

- `ta.ema` seeds with an SMA of the first `period` bars; the engine seeds
  with the first observation of the window. An EMA50 still carries ~9% of
  its seed after 60 bars.
- `ta.rsi` is Wilder's; the engine's is Cutler's (simple averages).
- `ta.atr` applies Wilder's smoothing; the engine takes a simple mean of
  true ranges.

The breakout range excludes the live bar, because a range that includes
the bar being tested moves with price and nothing ever breaks out.

The brain block runs on **daily** data via `request.security`, matching
the engine's daily bars plus live spot; VWAP and the opening range run on
the chart's own timeframe, matching the engine's intraday bars. Put it on
a 5-minute NIFTY chart and both halves are right.

A test parses the Pine and fails if its input defaults drift from
`IndexBrainConfig`, if it starts using one of the wrong built-ins, or if
it sends a trigger kind the receiver would refuse.

### What the panel says it cannot see

The panel carries a `NOT VISIBLE HERE: breadth · OI · IV · forward · VRP ·
greeks` row and a `decision: engine only — nothing here trades` row.
Without them a green BULLISH reads as the system's verdict rather than as
one of its four inputs.

### The one label that is not literal

A composite direction flip is the highest-value event on the chart and has
no trigger type of its own — the enum's chart-observable family is about
price events, not about this system's scores. It is sent as
`SIGNIFICANT_PRICE_MOVEMENT` with the composite and confidence in the
note. The engine re-derives the score from its own data on waking, so the
label decides only whether the pipeline is worth waking, never what it
finds.

## Setting it up

**1. Run the receiver.**

```bash
export TRADINGVIEW_WEBHOOK_SECRET="$(openssl rand -hex 24)"
python -m index_option_brain.integrations.tradingview
```

It refuses to start without a secret. An open POST endpoint on the public
internet that anyone who finds the URL can push market claims into is
worse than no receiver at all.

Behind a tunnel (Cloudflare, ngrok) the peer address is the tunnel, not
TradingView, so set `TRADINGVIEW_ALLOWED_IPS=any` — spelled out so it is a
decision rather than an oversight — and the secret becomes the only thing
between the endpoint and the internet.

**2. Add the indicator.** Paste `pine/index_brain_mirror.pine` into the
Pine editor, add it to a NIFTY or BANKNIFTY chart, set *Shared secret* to
the same value, and tick *Emit webhook alerts*.

The secret is an indicator input, so anyone you share the chart with — and
any screenshot of the settings dialog — has it. Use a secret dedicated to
this indicator and rotate it when you share a layout.

**3. Create one alert.** Condition **"Any alert() function call"**,
webhook URL `https://<host>:8787/tv/webhook`, message box **empty**. The
indicator builds the JSON itself: `alert()` does not substitute
`{{placeholders}}`, so every field including both timestamps is formatted
in Pine.

Webhooks require a paid TradingView plan; the free tier has none.

**4. Watch it.** `GET /api/tradingview` on the console (tailnet) lists
accepted alerts and `last_alert_at`. That age is the reading that matters:
a TradingView alert can stop firing without failing visibly — it expires,
or the account hits its alert limit — and from this side that is
indistinguishable from a quiet market.

## Symbols

Mapped exactly, never matched. `NIFTY`, `NIFTY1!`, `NSE:NIFTY`,
`BANKNIFTY`, `CNXBANK` and their prefixed forms; anything else is refused
rather than guessed. A substring rule would route every BANKNIFTY alert to
NIFTY with a plausible price attached, and nothing downstream could tell.
