"""Spec §31 — Observability Contract. Every trade must be reconstructable
from logs/metrics; this module pins the vocabulary so instrumentation added
across every stage stays consistent instead of ad hoc string metric names."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol


class MetricName(StrEnum):
    DATA_LATENCY = "data_latency"
    ANALYSIS_LATENCY = "analysis_latency"
    TRIGGER_COUNT = "trigger_count"
    SIGNAL_COUNT = "signal_count"
    TRADE_CANDIDATES = "trade_candidates"
    REJECTIONS = "rejections"
    RISK_FAILURES = "risk_failures"
    ORDERS = "orders"
    FILLS = "fills"
    POSITION_MISMATCHES = "position_mismatches"
    BROKER_ERRORS = "broker_errors"
    LLM_ERRORS = "llm_errors"
    SYSTEM_ERRORS = "system_errors"


class MetricsSink(Protocol):
    def increment(self, name: MetricName, *, amount: int = 1, **tags: str) -> None: ...

    def observe(self, name: MetricName, value: float, **tags: str) -> None: ...
