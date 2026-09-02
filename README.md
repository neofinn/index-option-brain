# index-option-brain

A production-grade, deterministic quantitative trading engine for Indian
index + index-options instruments. Built from a full architecture spec
("Indian Index + Options Brain — Master Architecture & Implementation
Contracts"). Futures are explicitly out of scope.

**Non-negotiable rule:** the system must run fully with `LLM_ENABLED=false`.
The deterministic quantitative brain is mandatory; the Risk Engine is
authoritative; the Execution Gate is mandatory; the LLM/agent layer
(`index_option_brain.agent`) is an optional investigation/reasoning add-on
that can never override risk, execution, position limits, or the kill
switch, and the engine must never require it to exist.

## Status

| Layer | State |
|---|---|
| Canonical Pydantic contracts (`contracts/`) | **Implemented** — `MarketState` and every brain/engine input/output |
| Data adapter interfaces (`data/adapters/base.py`) | **Implemented** — split by capability, not by vendor |
| `SimulatorDataAdapter` (`data/adapters/mock.py`) | **Implemented** — deterministic, Black-Scholes-priced, **not a live adapter** |
| Market-state engine (`state/`) | **Implemented** — assembly plus realized vol, ATM IV, sector aggregates, session state |
| Indicator library (`brain/indicators.py`) | **Implemented** — pure functions, no state, no I/O |
| Index Brain | **Implemented** — trend, swing structure, momentum, VWAP, ATR, S/R, breakout state, opening structure |
| Constituent Brain | **Implemented** — contribution, breadth, participation, leadership, concentration, sectors |
| Options Brain | **Implemented** — walls, gamma zones, OI pressure, PCR, skew, liquidity, max pain |
| Volatility Engine | **Implemented** — IV regime/percentile, IV-vs-realized richness, expected move, expansion |
| Regime Engine | **Implemented** — scored classification across all 12 regimes with a reachable UNCERTAIN |
| Scenario Engine | **Implemented** — competing futures, each with the evidence against it; NO_TRADE always generated |
| Signal Engine | **Implemented** — four gates: separation, agreement, primary-domain participation, conviction floor |
| Strategy Engine | **Implemented** — volatility-aware structure choice with real economics; NO_TRADE always valid |
| Strike Engine | **Implemented** — hard filters then multi-leg ranking on delta fit, liquidity, structure, walls |
| Position Brain | **Implemented** — thesis validity, P&L, lifecycle transitions, exitability |
| Analysis pipeline (`brain/pipeline.py`) | **Implemented** — spec §33 flow, MarketState → ranked contracts |
| Event / Trigger Engine (`events/`) | **Implemented** — 26 of 30 triggers detected; the other 4 are calendar facts with no source |
| Significance filter | **Implemented** — score floor plus per-trigger cooldown, so the engine stays quiet when the market is |
| Failure contract (§29) | **Implemented** — explicit domain→action mapping (`risk/failure_policy.py`) |
| `IntelligenceProvider` / `DeterministicProvider` / agent tools | **Implemented** — the deterministic provider is the always-on default |
| Risk Engine | **Implemented** — authoritative sizing from four budgets, fail-closed, no override path |
| Black-Scholes pricing / greeks (`analytics/`) | **Implemented** — production, because no Indian feed publishes greeks |
| **Live NSE adapter** (`data/adapters/nse_public.py`) | **Implemented** — index, expiries, full chain, India VIX |
| Live bar aggregator (`data/bar_aggregator.py`) | **Implemented** — builds bars from snapshots; NSE serves no history |
| Provider registry (`data/providers.py`) | **Implemented** — 11 providers, 1 with an adapter, the rest labelled |
| Execution Gate | **Implemented** — 16 blocking checks re-validated against the live market, no override path |
| Order Manager | **Implemented** — §30 state machine, sequenced multi-leg submission, naked-short detection, reconciliation |
| Broker adapter | Interface only — the one thing between analysis and trading |
| Feedback / learning engines | Interface only (`feedback/`) |
| Memory (Postgres repository, Redis cache) | Interface only (`memory/`) |
| Backtest/replay engine | Interface only (`backtest/`) |
| Database schema | `Base` + UUID/timestamp/version mixin only — the ~27 tables from §27 are not yet modeled |
| FastAPI app + operations console | **Implemented** — live status/providers/market/analysis endpoints, `docs/console.html` |

892 tests pass; `ruff` and `mypy --strict` are clean.

### Where the pipeline deliberately stops

The chain now runs end to end: `MarketState` → analysis → ranked structure →
`TradeCandidate` → `RiskDecision` → `TradeDecision` → Execution Gate →
`OrderRequest`. What it cannot do is **send** one, because no broker adapter
exists.

That boundary is enforced rather than documented. `QuantitativeBrain.run()`
only reaches risk when given an account and a portfolio, and without a broker
there is no account to give it — so `is_actionable` ("a candidate survived
analysis") stays distinct from `is_authorized` ("risk approved a size"), and
the second is currently always false. Inventing an account balance to make an
execution panel look populated is the single most dangerous placeholder
available in this system, and the API says "no broker connected" instead.

## Design decisions worth knowing

**Nothing trades on one indicator.** Every score is a blend of independent
measurements, and confidence falls when they disagree or when data is
missing. A directional signal additionally requires four independent gates to
pass — see `brain/signal_brain.py`. In particular the index domain must
itself vote for the direction, which is what structurally prevents open
interest from carrying a trade alone (spec §7).

**Missing data lowers confidence; it never becomes a default.** Indicators
return `None` rather than a fabricated reading, and brains degrade smoothly
via `indicators.blend`, which renormalizes over the components actually
present.

**Volatility decides the expression, not just the direction.** The same
bullish signal produces a put credit spread when premium is rich and a long
call or debit spread when it is cheap. Level (IV vs its own history) and
richness (IV vs realized) are separate fields because they answer separate
questions.

**Pricing is pessimistic.** Buys price at the ask, sells at the bid.
Mid-pricing a spread flatters every number, and the flattery compounds
through the Risk Engine.

**Structures are multi-leg and priced as a whole.** A spread's max loss,
breakeven, and capital are properties of the combination, so
`StrikeCandidate` carries legs rather than a single contract. The Strategy
and Strike engines share one implementation (`brain/structures.py`), so the
numbers a strategy was chosen on are the numbers the ranked contracts report.

**NO_TRADE competes on merit.** It is scored, not a fallback, and it outranks
any structure that fails to clear the acceptance floor — a weak candidate can
never win by being the only candidate.

**Time comes from the market, not the wall clock.** Position updates are
stamped from `MarketState.timestamp` so BACKTEST and REPLAY stay reproducible
(spec §22).

**Parameters are typed config, not magic numbers.** Every threshold lives in
`brain/config.py`, injected per brain — which is what a Learning Engine
proposal (spec §20) would eventually version.

## Running it always-on

The engine has to run continuously, not on page load: NSE serves no history,
so the only source of bars is the aggregator observing snapshots as the
session proceeds. `app/runner.py` polls on a loop, feeds each snapshot to the
Trigger Engine, and runs the brain **only when a significant event fires** —
which is the point of having an event engine rather than a timer. Measured on
a closed market: six cycles, one significant event, one analysis.

`deploy/` has a Dockerfile, a Compose file that serves the console over
`tailscale serve` (tailnet-only HTTPS, no host port published), and a
hardened systemd unit for a bare-metal install. See `deploy/README.md`.

Monitor `/ready`, not `/health`. `/health` is liveness and stays 200 through
a data outage on purpose — a process manager restarting the app because NSE
is rate-limiting would turn a data problem into an availability problem and
discard every bar observed that session. `/ready` answers 503 once the loop
has failed three polls in a row, because a process that is alive but blind
looks identical to a healthy one from `/health`.

## Operations console

`docs/console.html`, served at `/` by the FastAPI app. Start the app and open
it; it reads `/api/status`, `/api/providers`, `/api/market/{symbol}` and
`/api/analysis/{symbol}`.

**It contains no market figures of its own.** There is no sample payload and
no placeholder number in the markup — a test asserts that no real session's
figures have been pasted in. A value the API does not supply renders as an
explicit "not published"; a backend that cannot be reached renders a banner
saying so. A console showing plausible sample numbers is worse than one
showing none, because the first is indistinguishable from a working system.

Connections are a **dropdown**, not a matrix. An operator connects one
provider at a time, so only the selected provider's capabilities, credential
fields and caveats are shown. The list is grouped into "connectable now" and
"planned — no adapter yet", and capability chips render in three visually
distinct states:

- **verified** — a live call for it returned data
- **claimed** — declared by the provider's documentation, not yet proven
- **absent** — not served at all

That distinction is the point. Exactly one provider is implemented, and its
four capabilities were probed against the live endpoint; every other entry's
capability list is read off published documentation and is labelled as
unverified. A documented capability is a claim and a probed one is a fact, and
a trading system must not treat them alike.

### What the live feed actually serves

Probed against NSE's public API, not assumed:

| | |
|---|---|
| Index OHLC + previous close | works |
| India VIX | works (same request as the index) |
| Expiry list | works — weeklies are **Tuesdays**, not Thursdays |
| Full option chain: LTP, top-of-book, IV, OI, ΔOI, volume | works |
| Option greeks | **not published** — computed from premium and IV |
| Historical bars | **blocked** — the history endpoint serves an anti-bot page |
| Constituent quotes | **404** — index breadth needs another provider |
| Account / orders | not a broker |

Two consequences shaped the design:

- **Greeks are computed in-process** (`analytics/pricing.py`), which is why
  the Strike Engine can rank on delta fit at all.
- **Bars are aggregated from live snapshots** (`data/bar_aggregator.py`)
  until a broker adapter supplies history. With no bars there is no measured
  structure, and the Regime Engine correctly refuses to classify one — live,
  it reports UNCERTAIN with the reason.

### Implied volatility is marked to the book, not the last trade

NSE computes its published IV from the last traded price, and on a thin strike
that is a real problem. Measured on one live snapshot: the 22,900 CE published
**46.55%** IV off a trade at 1,190 while its book stood at 965.20/1,082.25 —
an IV that would have given the strike a delta of 0.78 and an enormous vega,
competing with genuine candidates in strike ranking. Meanwhile the 23,600 CE
had a book 1.20 wide on a 344 mid and NSE published **no** IV for it at all.

So the adapter marks to the mid of a book tight enough to mean something,
falls back to NSE's figure only when there is no markable book, and otherwise
reports nothing. Of 166 live legs that yields 113 with a usable IV and 53
honestly unmarkable, with near-ATM delta monotonic in strike across a 7–12%
smile. `prefer_published_iv=True` passes NSE's own numbers through for
reconciling against nseindia.com.

### Palette

`--up #55CE9A` / `--down #BC404C` are lightness-split on purpose: that split
earns a deuteranopia separation of ΔE 21 where a naive green/red pair scores
ΔE 7. Brass means "you must act", cyan means "live data", and there is
deliberately no yellow warning state because it would collide with brass.
India VIX is deliberately **not** coloured up/down — a falling VIX in green
beside a falling index in red reads as the two moving opposite ways, so its
direction is named in words instead.

## The failure the Order Manager exists for

A spread submitted leg by leg can end up with one leg on and one leg
rejected. If the leg that filled was the short one, the account is holding a
naked short option — unbounded risk, from a decision that authorized a
defined-risk spread — and it is invisible until a margin call.

Three things guard against it, in order:

1. The **Execution Gate** sequences the protective leg first, on
   `OrderRequest.sequence`. Indian brokers also grant spread margin only once
   the hedge is present, so buying first is both safer and cheaper.
2. The **Order Manager** re-sorts by sequence rather than trusting the caller
   (a safety property that depends on the caller is not a guarantee), stops
   sending the rest of a structure the moment a leg fails, and cancels
   whatever is still working — cancelling its own orders is unambiguously
   risk-reducing and needs no further authorization.
3. If short exposure is filled anyway without its hedge, the submission comes
   back with `unhedged_short=True`. Exposure is measured against what the
   **decision intended**, not against what was sent: a protective leg never
   submitted leaves the position just as naked as one that was rejected, and
   that is the more common case.

Remediation is a separate, explicit call (`flatten`). A flattening order is
one the Execution Gate never saw — its checks are all about opening risk, so
they do not apply to reducing it, but placing a trade the gate never
authorized is not something that layer should do on its own initiative. The
condition is reported; the remedy is invoked.

Two more silent failures the same layer covers: `CANCEL_PENDING → FILLED` is
a **legal** transition, because a cancel loses the race often enough that
treating it as impossible is how a system comes to believe it is flat while
holding a position; and submission is keyed on `client_order_id`, because a
cycle re-running before an acknowledgement arrives would otherwise double the
position, and afterwards the duplicate is indistinguishable from intent.

Order modification is refused rather than emulated. Cancel-and-replace is not
a modification: the replacement loses queue position and can be beaten to a
fill, so a caller believing it modified an order would be wrong about both its
price and its priority.

## Repository layout

```
index_option_brain/
├── app/            FastAPI app, live engine, console API
├── config/         Settings (LLM_ENABLED, RUN_MODE, DB/Redis URLs)
├── contracts/       Canonical Pydantic data contracts (spec §2-21)
├── analytics/       Black-Scholes pricing and greeks (production)
├── data/adapters/   Adapter interfaces, the live NSE adapter, the simulator
├── data/            HTTP seam, bar aggregator, provider registry
├── state/           Market-state engine (assembles MarketState)
├── events/          Trigger engine + significance filter (interfaces)
├── brain/           indicators, config, structures, the nine brains,
│                     position brain, and the analysis pipeline
├── risk/            Risk engine, limits, margin model, failure policy
├── execution/       Execution gate (implemented); order manager, broker (interfaces)
├── agent/           IntelligenceProvider, DeterministicProvider, agent tools
├── memory/          Postgres repository + Redis cache (interfaces)
├── feedback/        Feedback + learning engines (interfaces)
├── backtest/         Backtest/replay engine (interface)
├── database/        SQLAlchemy base + UUID/timestamp/version mixin
├── monitoring/       Observability metric names + sink protocol
└── tests/
```

## Getting started

Requires Python 3.12+.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
.venv/bin/ruff check .
.venv/bin/mypy index_option_brain
```

Copy `.env.example` to `.env` and adjust as needed. `LLM_ENABLED` defaults
to `false` and `RUN_MODE` defaults to `paper`.

### Running against the live feed

```bash
# Probe what NSE is serving right now, with every line measured.
.venv/bin/python scripts/live_nse_check.py NIFTY

# Start the API and open the console at http://127.0.0.1:8000/
.venv/bin/uvicorn index_option_brain.app.main:app
```

```python
import asyncio
from index_option_brain.brain import QuantitativeBrain
from index_option_brain.contracts.enums import BarInterval
from index_option_brain.data.adapters.nse_public import NsePublicAdapter
from index_option_brain.data.bar_aggregator import AggregatingIndexAdapter
from index_option_brain.state import InMemoryIvHistoryStore, MarketStateBuilder

async def main():
    async with NsePublicAdapter() as nse:
        # NSE serves no history, so it is wrapped: every snapshot read for
        # analysis doubles as a bar observation.
        index = AggregatingIndexAdapter(nse, intervals=(BarInterval.MINUTE_5, BarInterval.DAY))
        # `None` for constituents: no connected provider serves index breadth,
        # and the parameter has no default so the gap has to be stated.
        builder = MarketStateBuilder(index, None, nse, nse, InMemoryIvHistoryStore())
        state = await builder.build("NIFTY")

        result = QuantitativeBrain().run(state)
        print(result.regime.regime, result.signal.direction, result.selected_strategy)

asyncio.run(main())
```

On a cold start this prints `UNCERTAIN neutral NO_TRADE`, because there are no
bars yet and therefore no measured structure. That is the correct answer, not
a failure — leave the process running and the aggregator fills the history in.

### Running the brain against simulated data

```python
import asyncio
from index_option_brain.brain import QuantitativeBrain
from index_option_brain.data.adapters.mock import SimulatorDataAdapter
from index_option_brain.state import InMemoryIvHistoryStore, MarketStateBuilder

async def main():
    adapter = SimulatorDataAdapter(seed=7, daily_drift_pct=0.35, breadth_bias=0.6)
    builder = MarketStateBuilder(adapter, adapter, adapter, adapter, InMemoryIvHistoryStore())
    state = await builder.build("NIFTY")

    result = QuantitativeBrain().run(state)
    print(result.regime.regime, result.signal.direction, result.selected_strategy)
    if result.best_candidate:
        print(result.best_candidate.rationale)

asyncio.run(main())
```

The simulator's character is configurable (`daily_drift_pct`,
`mean_reversion`, `breadth_bias`, `heavyweight_bias`, `base_iv`), which is
how the tests construct genuine uptrends, ranges, and narrow
heavyweight-driven rallies and assert the brains read them correctly.

## Next steps

1. **A broker adapter.** The one thing standing between analysis and trading,
   and it also closes two data gaps: historical bars (which unblocks the
   Regime Engine) and an account (which unblocks the Risk Engine). Dhan's
   long-lived token or Angel One's TOTP login suit an unattended process
   better than the daily browser login the OAuth brokers require.
2. **Event/trigger engine** — real detection over consecutive `MarketState`
   snapshots, plus the significance filter.
4. **Postgres schema** for spec §27, then the feedback/learning pipeline.
5. **Backtest/replay engine** — the brains, the Risk Engine and the Execution
   Gate are all deterministic and clock-independent already, so this is mostly
   a data-source and simulated-fill exercise.
6. Optional `AIProvider` behind `IntelligenceProvider`, once there is
   something substantial for it to investigate.

### One number worth knowing before funding an account

At the default 1% risk-per-trade, a NIFTY put credit spread with a per-lot max
loss of about ₹11,835 needs roughly **₹12 lakh** of equity before the Risk
Engine will authorize a single lot. Below that, `BELOW_MINIMUM_SIZE` is the
correct and permanent answer.

## Source spec

Implemented from "Indian Index + Options Brain — Master Architecture &
Implementation Contracts" (36 sections covering core architecture, data
layer, market-state contract, event/trigger engine, all nine brains, risk,
execution gate, order manager, position engine, feedback/learning, memory,
backtest/replay, the optional LLM/agent contract, repository structure,
technology baseline, database/Redis contracts, the failure contract, state
machines, observability, and test requirements).
