"""Spec §4 — prevents unnecessary full analysis by gating which Events are
significant enough to invoke the Quantitative Brain pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod

from index_option_brain.contracts.events import Event


class SignificanceFilter(ABC):
    @abstractmethod
    def is_significant(self, event: Event) -> bool: ...
