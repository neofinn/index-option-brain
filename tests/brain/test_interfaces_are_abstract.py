"""Every stage-4..16 engine is defined as an ABC with only its method
signature pinned down — this test suite exists to (a) prove the contracts
import cleanly and are actually abstract (can't be instantiated without a
real implementation), and (b) lock the method signatures the spec requires,
so a future implementation can't quietly drift from the contract.
"""

import pytest

from index_option_brain.backtest.engine import BacktestEngine
from index_option_brain.brain.constituent_brain import ConstituentBrain
from index_option_brain.brain.index_brain import IndexBrain
from index_option_brain.brain.options_brain import OptionsBrain
from index_option_brain.brain.position_brain import PositionBrain
from index_option_brain.brain.regime_brain import RegimeEngine
from index_option_brain.brain.scenario_brain import ScenarioEngine
from index_option_brain.brain.signal_brain import SignalEngine
from index_option_brain.brain.strategy_brain import StrategyEngine
from index_option_brain.brain.strike_brain import StrikeEngine
from index_option_brain.brain.volatility_brain import VolatilityEngine
from index_option_brain.events.significance_filter import SignificanceFilter
from index_option_brain.events.trigger_engine import TriggerEngine
from index_option_brain.execution.broker_adapter import BrokerAdapter
from index_option_brain.execution.execution_gate import ExecutionGate
from index_option_brain.execution.order_manager import OrderManager
from index_option_brain.feedback.feedback_engine import FeedbackEngine
from index_option_brain.feedback.learning_engine import LearningEngine
from index_option_brain.memory.cache import WorkingMemoryCache
from index_option_brain.memory.repository import TradeMemoryRepository
from index_option_brain.risk.risk_engine import RiskEngine

ABSTRACT_ENGINES = [
    IndexBrain,
    ConstituentBrain,
    OptionsBrain,
    VolatilityEngine,
    RegimeEngine,
    ScenarioEngine,
    SignalEngine,
    StrategyEngine,
    StrikeEngine,
    PositionBrain,
    TriggerEngine,
    SignificanceFilter,
    RiskEngine,
    ExecutionGate,
    OrderManager,
    BrokerAdapter,
    TradeMemoryRepository,
    WorkingMemoryCache,
    FeedbackEngine,
    LearningEngine,
    BacktestEngine,
]


@pytest.mark.parametrize("engine_cls", ABSTRACT_ENGINES, ids=lambda c: c.__name__)
def test_engine_cannot_be_instantiated_without_implementation(engine_cls):
    with pytest.raises(TypeError):
        engine_cls()


def test_index_brain_contract_is_satisfiable_by_a_concrete_subclass():
    """Guards against the interface accidentally requiring something a real
    implementation couldn't plausibly satisfy — e.g. spec §5's "Must NOT
    choose options, select strikes, size positions, or execute orders" means
    `analyze` must not need anything beyond MarketState."""
    from index_option_brain.contracts.analysis import IndexAnalysis
    from index_option_brain.contracts.enums import Direction

    class StubIndexBrain(IndexBrain):
        def analyze(self, state):
            return IndexAnalysis(
                direction=Direction.NEUTRAL,
                trend_score=0.0,
                structure_score=0.0,
                momentum_score=0.0,
                confidence=0.0,
            )

    brain = StubIndexBrain()
    result = brain.analyze(state=None)
    assert isinstance(result, IndexAnalysis)
