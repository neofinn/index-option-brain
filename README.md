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

This is the foundational scaffold. What's real vs. what's a typed contract
waiting for logic:

| Layer | State |
|---|---|
| Canonical Pydantic contracts (`contracts/`) | Implemented — `MarketState` and every brain/engine input/output shape from the spec |
| Data adapter interfaces (`data/adapters/base.py`) | Implemented |
| `SimulatorDataAdapter` (`data/adapters/mock.py`) | Implemented — deterministic, seeded, **not a live adapter** |
| Market-state engine (`state/market_state_builder.py`) | Implemented — assembles adapter output into `MarketState` |
| Event/trigger engine, significance filter | Interface only (`events/`) |
| Index / Constituent / Options / Volatility brains | Interface only (`brain/`) |
| Regime / Scenario / Signal / Strategy / Strike engines | Interface only (`brain/`) |
| Position brain | Interface only (`brain/position_brain.py`) |
| Risk engine | Interface only (`risk/risk_engine.py`) |
| Failure contract (§29) | Implemented as an explicit domain→action mapping (`risk/failure_policy.py`) |
| Execution gate / order manager / broker adapter | Interface only (`execution/`) |
| Position/feedback/learning engines | Interface only (`feedback/`) |
| Memory (Postgres repository, Redis cache) | Interface only (`memory/`) |
| Database base (UUID/timestamp/version mixin) | Implemented; the ~27 tables from spec §27 are **not yet modeled** — see below |
| Backtest/replay engine | Interface only (`backtest/`) |
| `IntelligenceProvider` / `DeterministicProvider` / agent tools | Implemented — `DeterministicProvider` is the always-available no-op default |
| Monitoring vocabulary | Implemented (`monitoring/metrics.py`) — no sink wired up yet |
| FastAPI app | Minimal — `/health` only |

Every interface-only module is an `ABC` with the exact method signature the
spec calls for, so a real implementation slots in without renegotiating the
contract, and `tests/brain/test_interfaces_are_abstract.py` fails loudly if
a signature drifts.

**Deliberately not modeled yet:** the full SQLAlchemy schema for spec §27's
~27 tables. Guessing column-level design before the query patterns from a
real brain/risk/execution implementation exist would just mean redoing it;
`database/base.py` only fixes the shared `Base` + `TimestampedUUIDMixin` so
that work is additive.

## Why interfaces first

Sections 25 and 36 of the spec ask for real typed Python — not a diagram —
built in a specific incremental order (scaffold → contracts → adapters →
market-state engine → event engine → brains → regime/scenario/signal →
strategy/strike → risk → execution/broker → position → feedback/memory →
backtest → providers → monitoring). This scaffold covers stages 1–4 as
working code and stages 5–16 as typed, testable interfaces — each with the
spec's own invariants encoded as docstrings and, where practical, as tests
(e.g. `NO_TRADE` is always constructible; every `FailureDomain` maps to a
safe action; `AgentAssessment` structurally can't carry execution
authority). Implementing real decision logic for nine analysis brains, a
risk engine, and broker execution in one pass — without real market data or
a chosen broker — would mean guessing at logic that has to be rebuilt once
it's tested against actual data anyway.

## Repository layout

```
index_option_brain/
├── app/            FastAPI app factory
├── config/         Settings (LLM_ENABLED, RUN_MODE, DB/Redis URLs)
├── contracts/       Canonical Pydantic data contracts (spec §2-21)
├── data/adapters/   Provider-agnostic data adapter interfaces + simulator
├── state/           Market-state engine (assembles MarketState)
├── events/          Trigger engine + significance filter (interfaces)
├── brain/           Index/Constituent/Options/Volatility/Regime/Scenario/
│                     Signal/Strategy/Strike/Position brains (interfaces)
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

## Next steps

Roughly in spec order:
1. Event/trigger engine + significance filter — real detection logic over
   consecutive `MarketState` snapshots.
2. Index/Constituent/Options/Volatility brains — the actual analysis logic.
3. Regime/Scenario/Signal engines.
4. Strategy/Strike engines.
5. Risk engine — the first stage that needs to be genuinely hardened and
   heavily tested (spec §32: "Risk and execution modules require especially
   strong coverage").
6. Execution gate / order manager / a real broker adapter (provider TBD —
   the spec's adapters are provider-agnostic by design).
7. Position engine, feedback/learning, Postgres schema for spec §27, backtest
   engine.
8. Optional `AIProvider` behind `IntelligenceProvider` once everything above
   exists to investigate.

## Source spec

Implemented from "Indian Index + Options Brain — Master Architecture &
Implementation Contracts" (36 sections covering core architecture, data
layer, market-state contract, event/trigger engine, all nine brains, risk,
execution gate, order manager, position engine, feedback/learning, memory,
backtest/replay, the optional LLM/agent contract, repository structure,
technology baseline, database/Redis contracts, the failure contract, state
machines, observability, and test requirements).
