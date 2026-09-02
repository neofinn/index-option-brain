"""Spec §24 — the only surface an enabled AI agent may call. Every method is
read-only. Do not expose unrestricted database access from here or anywhere
else the agent can reach."""

from __future__ import annotations

from abc import ABC, abstractmethod

from index_option_brain.contracts.analysis import (
    ConstituentAnalysis,
    IndexAnalysis,
    OptionsAnalysis,
    RegimeState,
    VolatilityAnalysis,
)
from index_option_brain.contracts.market_state import MarketState
from index_option_brain.contracts.scenario import Scenario


class AgentTools(ABC):
    @abstractmethod
    async def get_market_state(self) -> MarketState: ...

    @abstractmethod
    async def get_index_analysis(self) -> IndexAnalysis: ...

    @abstractmethod
    async def get_constituent_analysis(self) -> ConstituentAnalysis: ...

    @abstractmethod
    async def get_sector_analysis(self) -> dict[str, float]: ...

    @abstractmethod
    async def get_option_analysis(self) -> OptionsAnalysis: ...

    @abstractmethod
    async def get_volatility_analysis(self) -> VolatilityAnalysis: ...

    @abstractmethod
    async def get_regime(self) -> RegimeState: ...

    @abstractmethod
    async def get_scenarios(self) -> list[Scenario]: ...

    @abstractmethod
    async def get_trade_history(self, limit: int = 20) -> list[dict[str, object]]: ...

    @abstractmethod
    async def get_similar_scenarios(self, scenario_id: str, limit: int = 5) -> list[Scenario]: ...

    @abstractmethod
    async def challenge_thesis(self, thesis_id: str) -> str: ...
