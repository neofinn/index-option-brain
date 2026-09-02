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
| Failure contract (§29) | **Implemented** — explicit domain→action mapping (`risk/failure_policy.py`) |
| `IntelligenceProvider` / `DeterministicProvider` / agent tools | **Implemented** — the deterministic provider is the always-on default |
| Risk Engine | Interface only (`risk/risk_engine.py`) |
| Execution gate / order manager / broker adapter | Interface only (`execution/`) |
| Feedback / learning engines | Interface only (`feedback/`) |
| Memory (Postgres repository, Redis cache) | Interface only (`memory/`) |
| Backtest/replay engine | Interface only (`backtest/`) |
| Database schema | `Base` + UUID/timestamp/version mixin only — the ~27 tables from §27 are not yet modeled |
| FastAPI app | Minimal — `/health` only |

338 tests pass; `ruff` and `mypy --strict` are clean.

### Where the pipeline deliberately stops

`QuantitativeBrain.run()` ends at **ranked strike candidates**. It does not
construct a `TradeDecision`, because that would require inventing a
`RiskDecision` — and a fabricated risk approval is exactly the
placeholder-as-production that spec §36 prohibits.
`BrainCycleResult.is_actionable` means "a candidate survived analysis", never
"this is authorized".

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

## Operations console design

`docs/console-design.html` is a self-contained design for the operating
surface: every broker and data-feed connection option with its capability
coverage, credential shape and token lifetime; the run-mode / LLM / kill-switch
controls; and the brain's current decision chain through to the execution gate.
Open it directly in a browser.

Two findings from building the connection matrix are worth stating here,
because they shape the adapter work:

- **No Indian broker API returns Greeks.** NSE's chain endpoint publishes
  implied volatility but no sensitivities. Delta, gamma, theta and vega are
  therefore computed in-process — which is why the Strike Engine can rank on
  delta fit at all.
- **Token lifetime is an architectural constraint, not a detail.** Most broker
  tokens expire at the next login window, so an unattended engine needs either
  a long-lived token or a login that can be scripted from a stored TOTP secret.

The numbers shown in that page are a real run of the implemented pipeline
against the simulator, not illustrative placeholders.

## Repository layout

```
index_option_brain/
├── app/            FastAPI app factory
├── config/         Settings (LLM_ENABLED, RUN_MODE, DB/Redis URLs)
├── contracts/       Canonical Pydantic data contracts (spec §2-21)
├── data/adapters/   Provider-agnostic adapter interfaces + simulator
├── state/           Market-state engine (assembles MarketState)
├── events/          Trigger engine + significance filter (interfaces)
├── brain/           indicators, config, structures, the nine brains,
│                     position brain, and the analysis pipeline
├── risk/            Risk engine (interface) + failure policy (implemented)
├── execution/       Execution gate / order manager / broker adapter (interfaces)
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

1. **Risk Engine** — the first stage that must be genuinely hardened
   (spec §32: "Risk and execution modules require especially strong
   coverage"). Position sizing, exposure, daily loss limits, concentration.
2. **Execution gate + order manager + a real broker adapter** (provider TBD —
   the adapters are provider-agnostic by design).
3. **Event/trigger engine** — real detection over consecutive MarketState
   snapshots, plus the significance filter.
4. **Postgres schema** for spec §27, then the feedback/learning pipeline.
5. **Backtest/replay engine** — the brains are already deterministic and
   clock-independent, so this is mostly a data-source and simulated-fill
   exercise.
6. Optional `AIProvider` behind `IntelligenceProvider`, once there is
   something substantial for it to investigate.

## Source spec

Implemented from "Indian Index + Options Brain — Master Architecture &
Implementation Contracts" (36 sections covering core architecture, data
layer, market-state contract, event/trigger engine, all nine brains, risk,
execution gate, order manager, position engine, feedback/learning, memory,
backtest/replay, the optional LLM/agent contract, repository structure,
technology baseline, database/Redis contracts, the failure contract, state
machines, observability, and test requirements).
