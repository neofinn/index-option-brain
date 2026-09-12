"""Spec §28 — Redis as working memory/cache/event streaming. PostgreSQL
remains the persistent source of truth; this is scratch/hot-path state only
(latest MarketState/OptionsState, active PositionState, event stream, locks,
dedup state) and must fail safe (spec §29) rather than silently serving
stale data on a Redis outage."""

from __future__ import annotations

from abc import ABC, abstractmethod

from index_option_brain.contracts.market_state import MarketState


class WorkingMemoryCache(ABC):
    @abstractmethod
    async def set_latest_market_state(self, state: MarketState) -> None: ...

    @abstractmethod
    async def get_latest_market_state(self) -> MarketState | None: ...

    @abstractmethod
    async def acquire_lock(self, key: str, ttl_seconds: int) -> bool: ...

    @abstractmethod
    async def release_lock(self, key: str) -> None: ...

    @abstractmethod
    async def seen_before(self, dedup_key: str) -> bool:
        """Idempotency/deduplication check, e.g. for duplicate-order
        prevention (spec §16 `duplicate_order_check`)."""
        ...
