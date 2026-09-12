"""The quantitative brain pipeline — spec §33's master decision flow, from
MarketState through to ranked, executable contracts.

    MARKET STATE
      -> INDEX + CONSTITUENT + OPTIONS + VOLATILITY ANALYSIS
      -> REGIME
      -> SCENARIOS
      -> SIGNAL
      -> STRATEGY CANDIDATES
      -> STRIKE CANDIDATES
      -> (trade candidate)

Risk authorization runs when — and only when — an account and portfolio are
supplied. Given them, the cycle produces a real `TradeDecision`: EXECUTE if
risk authorized a size, REJECT if it declined, WAIT if nothing reached risk.
Without them the cycle stops at ranked candidates and no decision is
constructed, because a `TradeDecision` requires a `RiskDecision` and this
pipeline will not fabricate one (spec §36).

The flow still stops short of an order. `TradeDecision.is_executable` means
risk authorized a size; the Execution Gate (§16) and Order Manager (§17) are
what turn that into an order, and neither is implemented.

Every stage is injected, so a future strategy version can swap one brain
without touching the others, and each stage's output is folded back into the
(immutable) MarketState so downstream stages read state rather than
positional arguments.
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict, Field

from index_option_brain.brain.constituent_brain import (
    ConstituentBrain,
    DeterministicConstituentBrain,
)
from index_option_brain.brain.index_brain import DeterministicIndexBrain, IndexBrain
from index_option_brain.brain.options_brain import DeterministicOptionsBrain, OptionsBrain
from index_option_brain.brain.position_brain import DeterministicPositionBrain, PositionBrain
from index_option_brain.brain.regime_brain import DeterministicRegimeEngine, RegimeEngine
from index_option_brain.brain.scenario_brain import (
    DeterministicScenarioEngine,
    ScenarioEngine,
)
from index_option_brain.brain.signal_brain import DeterministicSignalEngine, SignalEngine
from index_option_brain.brain.strategy_brain import (
    DeterministicStrategyEngine,
    StrategyEngine,
)
from index_option_brain.brain.strike_brain import DeterministicStrikeEngine, StrikeEngine
from index_option_brain.brain.volatility_brain import (
    DeterministicVolatilityEngine,
    VolatilityEngine,
)
from index_option_brain.contracts.analysis import AnalysisBundle, RegimeState
from index_option_brain.contracts.decision import TradeDecision
from index_option_brain.contracts.enums import Direction, StrategyType
from index_option_brain.contracts.instruments import AccountSnapshot
from index_option_brain.contracts.market_state import MarketState
from index_option_brain.contracts.position import Position, PositionState
from index_option_brain.contracts.risk import PortfolioState, RiskDecision, TradeCandidate
from index_option_brain.contracts.scenario import Scenario
from index_option_brain.contracts.signal import Signal
from index_option_brain.contracts.strategy import StrategyCandidate
from index_option_brain.contracts.strike import StrikeCandidate
from index_option_brain.risk.risk_engine import DeterministicRiskEngine, RiskEngine


class BrainCycleResult(BaseModel):
    """Everything one analysis cycle produced, in one auditable object.

    Spec §31 requires every trade to be reconstructable from logs; keeping
    the whole chain of reasoning — not just the conclusion — is what makes
    that possible.
    """

    model_config = ConfigDict(frozen=True)

    state: MarketState
    analysis: AnalysisBundle
    regime: RegimeState
    scenarios: list[Scenario] = Field(default_factory=list)
    signal: Signal
    strategy_candidates: list[StrategyCandidate] = Field(default_factory=list)
    strike_candidates: list[StrikeCandidate] = Field(default_factory=list)
    selected_strategy: StrategyType = StrategyType.NO_TRADE
    positions: list[Position] = Field(default_factory=list)
    candidate: TradeCandidate | None = None
    risk_decision: RiskDecision | None = None
    decision: TradeDecision | None = None

    @property
    def is_actionable(self) -> bool:
        """A candidate survived analysis. NOT an authorization — see
        `is_authorized` for that."""
        return (
            self.selected_strategy is not StrategyType.NO_TRADE
            and bool(self.strike_candidates)
        )

    @property
    def is_authorized(self) -> bool:
        """Risk approved a size. Still not an order: the Execution Gate has
        not run, and it re-validates independently."""
        return self.decision is not None and self.decision.is_executable

    @property
    def best_candidate(self) -> StrikeCandidate | None:
        return self.strike_candidates[0] if self.strike_candidates else None


class QuantitativeBrain:
    """Runs the deterministic analysis chain. Contains no LLM call and no
    broker access, and requires neither to function (spec §23, §35)."""

    def __init__(
        self,
        *,
        index_brain: IndexBrain | None = None,
        constituent_brain: ConstituentBrain | None = None,
        options_brain: OptionsBrain | None = None,
        volatility_engine: VolatilityEngine | None = None,
        regime_engine: RegimeEngine | None = None,
        scenario_engine: ScenarioEngine | None = None,
        signal_engine: SignalEngine | None = None,
        strategy_engine: StrategyEngine | None = None,
        strike_engine: StrikeEngine | None = None,
        position_brain: PositionBrain | None = None,
        risk_engine: RiskEngine | None = None,
    ) -> None:
        self._index_brain = index_brain or DeterministicIndexBrain()
        self._constituent_brain = constituent_brain or DeterministicConstituentBrain()
        self._options_brain = options_brain or DeterministicOptionsBrain()
        self._volatility_engine = volatility_engine or DeterministicVolatilityEngine()
        self._regime_engine = regime_engine or DeterministicRegimeEngine()
        self._scenario_engine = scenario_engine or DeterministicScenarioEngine()
        self._signal_engine = signal_engine or DeterministicSignalEngine()
        self._strategy_engine = strategy_engine or DeterministicStrategyEngine()
        self._strike_engine = strike_engine or DeterministicStrikeEngine()
        self._position_brain = position_brain or DeterministicPositionBrain()
        self._risk_engine = risk_engine or DeterministicRiskEngine()

    def run(
        self,
        state: MarketState,
        *,
        account: AccountSnapshot | None = None,
        portfolio: PortfolioState | None = None,
    ) -> BrainCycleResult:
        """Run one analysis cycle.

        Supplying `account` and `portfolio` runs risk authorization and
        produces a `TradeDecision`. Without them the cycle stops at ranked
        candidates: a decision cannot be constructed without a real
        `RiskDecision`, and this pipeline will not invent one.
        """
        analysis = AnalysisBundle(
            index=self._index_brain.analyze(state),
            constituents=self._constituent_brain.analyze(state),
            options=self._options_brain.analyze(state),
            volatility=self._volatility_engine.analyze(state),
        )
        state = state.with_analysis(analysis)

        regime = self._regime_engine.classify(
            analysis.index, analysis.constituents, analysis.options, analysis.volatility
        )
        state = state.with_regime(regime)

        scenarios = self._scenario_engine.generate(state, regime)
        state = state.with_scenarios(scenarios)

        signal = self._signal_engine.evaluate(state, scenarios)
        state = state.with_signals([signal])

        strategy_candidates = self._strategy_engine.select(state, signal)
        selected = strategy_candidates[0].strategy if strategy_candidates else StrategyType.NO_TRADE

        strike_candidates: list[StrikeCandidate] = []
        if selected is not StrategyType.NO_TRADE:
            strike_candidates = self._strike_engine.rank(
                selected, state.options_state.chain, state
            )
            if not strike_candidates:
                # No contract survived the hard filters, so the structure the
                # Strategy Engine chose is not actually expressible right now.
                selected = StrategyType.NO_TRADE

        positions = [
            self._position_brain.evaluate(position, state)
            for position in state.position_state.positions
        ]
        if positions:
            state = state.with_position_state(PositionState(positions=positions))

        candidate = self._build_candidate(state, signal, selected, strike_candidates)
        risk_decision, decision = self._authorize(state, candidate, account, portfolio)

        return BrainCycleResult(
            state=state,
            analysis=analysis,
            regime=regime,
            scenarios=scenarios,
            signal=signal,
            strategy_candidates=strategy_candidates,
            strike_candidates=strike_candidates,
            selected_strategy=selected,
            positions=positions,
            candidate=candidate,
            risk_decision=risk_decision,
            decision=decision,
        )

    def _build_candidate(
        self,
        state: MarketState,
        signal: Signal,
        selected: StrategyType,
        strike_candidates: list[StrikeCandidate],
    ) -> TradeCandidate | None:
        """Assemble spec §33's TRADE CANDIDATE from the best surviving
        contract. The thesis_id minted here is what follows the trade through
        risk, execution, the position, and its eventual feedback (§15)."""
        if selected is StrategyType.NO_TRADE or not strike_candidates:
            return None

        best = strike_candidates[0]
        return TradeCandidate(
            state_id=state.state_id,
            thesis_id=uuid.uuid4().hex[:12],
            signal_id=signal.signal_id,
            scenario_id=signal.selected_scenario_id,
            direction=signal.direction,
            strategy=selected,
            structure=best,
            underlying_symbol=state.index_symbol,
            confidence=signal.confidence,
            entry_conditions=[
                f"{selected.value} at {best.rationale.split(':')[0]}"
                if ":" in best.rationale
                else selected.value
            ],
            target_conditions=(
                [f"Structure reaches its maximum profit of {best.max_profit}"]
                if best.max_profit is not None
                else ["Managed against the expected move; profit is unbounded"]
            ),
            invalidation_conditions=list(signal.invalidation_conditions),
            evidence=list(signal.evidence),
        )

    def _authorize(
        self,
        state: MarketState,
        candidate: TradeCandidate | None,
        account: AccountSnapshot | None,
        portfolio: PortfolioState | None,
    ) -> tuple[RiskDecision | None, TradeDecision | None]:
        """Risk runs only with real account and portfolio state.

        Without them there is nothing to authorize *against*, and inventing a
        default account would produce an approval that means nothing.
        """
        if account is None or portfolio is None:
            return None, None

        if candidate is None:
            return None, TradeDecision.wait(
                state_id=state.state_id,
                reasons=["No candidate survived analysis; standing aside"],
                timestamp=state.timestamp,
                direction=Direction.NEUTRAL,
            )

        risk_decision = self._risk_engine.authorize(candidate, account, portfolio)
        decision = TradeDecision.from_candidate(
            candidate, risk_decision, timestamp=state.timestamp
        )
        return risk_decision, decision
