"""The quantitative brain.

Each module exposes the spec's abstract contract (`IndexBrain`,
`RegimeEngine`, ...) alongside a concrete `Deterministic*` implementation.
The abstractions are what the rest of the system depends on; the
deterministic implementations are the default, and they contain no LLM call
and no broker access.
"""

from index_option_brain.brain.config import BrainConfig
from index_option_brain.brain.constituent_brain import (
    ConstituentBrain,
    DeterministicConstituentBrain,
)
from index_option_brain.brain.index_brain import DeterministicIndexBrain, IndexBrain
from index_option_brain.brain.options_brain import DeterministicOptionsBrain, OptionsBrain
from index_option_brain.brain.pipeline import BrainCycleResult, QuantitativeBrain
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

__all__ = [
    "BrainConfig",
    "BrainCycleResult",
    "ConstituentBrain",
    "DeterministicConstituentBrain",
    "DeterministicIndexBrain",
    "DeterministicOptionsBrain",
    "DeterministicPositionBrain",
    "DeterministicRegimeEngine",
    "DeterministicScenarioEngine",
    "DeterministicSignalEngine",
    "DeterministicStrategyEngine",
    "DeterministicStrikeEngine",
    "DeterministicVolatilityEngine",
    "IndexBrain",
    "OptionsBrain",
    "PositionBrain",
    "QuantitativeBrain",
    "RegimeEngine",
    "ScenarioEngine",
    "SignalEngine",
    "StrategyEngine",
    "StrikeEngine",
    "VolatilityEngine",
]
