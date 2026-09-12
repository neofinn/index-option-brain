"""SQLAlchemy declarative base and the mixin every persisted object needs
(spec §27: "Every important object needs: UUID, created_at, updated_at,
version").

The full table set from spec §27 (indices, constituents, constituent_weights,
market_snapshots, option_contracts, option_snapshots, events, index_analyses,
constituent_analyses, option_analyses, volatility_analyses, regimes,
scenarios, signals, strategy_candidates, strike_candidates, trade_decisions,
risk_decisions, orders, order_events, positions, position_events,
trade_feedback, lessons, strategy_versions, system_events) is deliberately
NOT modeled yet — column-level schema design should follow the real query
patterns established once the brain/risk/execution stages are implemented,
not be guessed upfront. This module only fixes the shared base so that work
is additive later.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class TimestampedUUIDMixin:
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    version: Mapped[int] = mapped_column(default=1)
