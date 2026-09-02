"""Spec §4 — Event / Trigger Engine.

Interface only: detecting each TriggerType requires comparing successive
MarketState snapshots (and, for time triggers, wall-clock/session context)
using thresholds that are themselves a tuning exercise, not a one-shot
scaffolding task. Wiring this up is the next implementation stage.

Contract invariant that must hold however this is implemented: a trigger
only means "something changed; analyze it" — it must NEVER directly create
an order, and must go through the SignificanceFilter before waking the full
brain pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from index_option_brain.contracts.events import Event
from index_option_brain.contracts.market_state import MarketState


class TriggerEngine(ABC):
    @abstractmethod
    def detect(self, previous_state: MarketState | None, current_state: MarketState) -> list[Event]:
        """Compare two consecutive MarketState snapshots (previous_state may be
        None on the first tick) and return the Events that fired."""
        ...
