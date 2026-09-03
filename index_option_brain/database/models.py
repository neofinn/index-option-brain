"""Spec §27 tables, modelled now that the query patterns exist.

`base.py` deliberately deferred this: column-level schema should follow the
queries the system actually makes, not a guess. Those queries now exist, and
they are what shaped this file:

* **The dashboard wants a session's history** — spot, regime, signal, strategy
  over the day. That is `market_snapshots` joined to `analysis_cycles`.
* **The replay wants option chains it does not have.** NSE serves no chain
  history and nothing free does, so P&L cannot be backtested — unless the
  system records what it sees while it runs. `option_snapshots` exists to
  turn every day of uptime into a day of future backtest corpus, which is
  the highest-value thing persistence buys here.
* **Spec §31 wants every trade reconstructable.** `analysis_cycles` keeps the
  whole chain of reasoning — evidence lists included — not just the verdict.
* **The learning pipeline wants trade outcomes and the lessons drawn from
  them**, kept apart so a lesson can never be mistaken for a live parameter.

Tables for orders and positions are still absent, and deliberately: the
broker adapter's response mapping has not been verified against live
payloads, so their columns would be a guess about a payload shape nobody
has seen. They arrive with that verification.

Portability
-----------
Types are chosen so the same models run on PostgreSQL (production, per spec
§21) and on SQLite (a fresh VPS, and the tests). `base.py` pins a
Postgres-only UUID column, so this module defines its own portable variants
rather than changing a shared base other code already depends on: a box
should start persisting before someone installs a database server, and a
schema that only exists under Postgres would mean the first weeks of chain
capture — the irreplaceable part — never happen.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, ClassVar

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator


class PortableUUID(TypeDecorator[uuid.UUID]):
    """A UUID column that is native on PostgreSQL and CHAR(36) elsewhere.

    Stored as text on SQLite rather than as bytes so a captured database can
    be read with a plain `sqlite3` shell — the corpus is meant to outlive
    this codebase.
    """

    impl = String(36)
    cache_ok = True

    def load_dialect_impl(self, dialect: Any) -> Any:
        if dialect.name == "postgresql":
            from sqlalchemy.dialects.postgresql import UUID as PgUUID

            return dialect.type_descriptor(PgUUID(as_uuid=True))
        return dialect.type_descriptor(String(36))

    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
        return str(value)

    def process_result_value(self, value: Any, dialect: Any) -> uuid.UUID | None:
        if value is None:
            return None
        return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


#: Prices are money. Float would round 23,914.45 into something that is not
#: 23,914.45, and the whole system compares levels for equality.
Price = Numeric(14, 4)


class Base(DeclarativeBase):
    # SQLAlchemy reads this off the class, so it is a ClassVar rather than a
    # mapped attribute.
    type_annotation_map: ClassVar[dict[Any, Any]] = {
        dict[str, Any]: JSON,
        list[str]: JSON,
    }


class Recorded:
    """Spec §27's shared columns: UUID, created_at, updated_at, version."""

    id: Mapped[uuid.UUID] = mapped_column(
        PortableUUID, primary_key=True, default=uuid.uuid4
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(),
        nullable=False,
    )
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)


class MarketSnapshot(Base, Recorded):
    """One observed market state, keyed by the **feed's** timestamp.

    `observed_at` is the moment the exchange published, not the moment this
    row was written — `created_at` is that. Keeping both is what lets a
    delayed or replayed snapshot be told apart from a live one after the
    fact, and the unique constraint on (symbol, observed_at) makes the
    capture idempotent: re-running it over a day it already holds is a
    no-op rather than a second copy.
    """

    __tablename__ = "market_snapshots"
    __table_args__ = (
        UniqueConstraint("index_symbol", "observed_at", name="uq_snapshot_moment"),
        Index("ix_snapshot_symbol_time", "index_symbol", "observed_at"),
    )

    index_symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    session_state: Mapped[str] = mapped_column(String(24), nullable=False)

    spot: Mapped[Any] = mapped_column(Price, nullable=False)
    day_open: Mapped[Any] = mapped_column(Price, nullable=True)
    day_high: Mapped[Any] = mapped_column(Price, nullable=True)
    day_low: Mapped[Any] = mapped_column(Price, nullable=True)
    previous_close: Mapped[Any] = mapped_column(Price, nullable=True)

    # Every one of these is nullable because every one can be genuinely
    # unmeasured, and a zero would read as a measurement of zero.
    india_vix: Mapped[float | None] = mapped_column(Float, nullable=True)
    atm_iv: Mapped[float | None] = mapped_column(Float, nullable=True)
    expiry: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    forward: Mapped[Any] = mapped_column(Price, nullable=True)
    forward_excess_basis: Mapped[Any] = mapped_column(Price, nullable=True)
    forward_strikes_used: Mapped[int | None] = mapped_column(Integer, nullable=True)

    breadth_advances: Mapped[int | None] = mapped_column(Integer, nullable=True)
    breadth_declines: Mapped[int | None] = mapped_column(Integer, nullable=True)
    breadth_observed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    """When the breadth reading was taken, which is not `observed_at`.

    Breadth comes from the opening auction and is frozen after 09:12, so a
    single timestamp on the row would make a 09:07 reading look as fresh as
    the 14:00 spot beside it.
    """

    option_legs: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    cycles: Mapped[list[AnalysisCycle]] = relationship(
        back_populates="snapshot", cascade="all, delete-orphan"
    )
    options: Mapped[list[OptionSnapshot]] = relationship(
        back_populates="snapshot", cascade="all, delete-orphan"
    )


class OptionSnapshot(Base, Recorded):
    """One leg of one chain, at one moment. The backtest corpus.

    This is the only table here whose value comes from sheer accumulation.
    Nothing free serves historical Indian option chains, so every day this
    runs is a day of P&L backtesting that becomes possible later and cannot
    be recovered if missed.

    Greeks are stored alongside the premium because they were computed
    against a *forward that will not be recoverable later* — the parity
    solve needs the whole book, and only the mid survives here. Recomputing
    them from the stored premium against a spot-derived forward would give
    different numbers than the system actually acted on.
    """

    __tablename__ = "option_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "snapshot_id", "strike", "option_type", name="uq_option_leg"
        ),
        Index("ix_option_expiry_strike", "expiry", "strike", "option_type"),
    )

    snapshot_id: Mapped[uuid.UUID] = mapped_column(
        PortableUUID, ForeignKey("market_snapshots.id", ondelete="CASCADE"),
        nullable=False,
    )
    underlying_symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    expiry: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    strike: Mapped[Any] = mapped_column(Price, nullable=False)
    option_type: Mapped[str] = mapped_column(String(2), nullable=False)

    ltp: Mapped[Any] = mapped_column(Price, nullable=True)
    bid: Mapped[Any] = mapped_column(Price, nullable=True)
    ask: Mapped[Any] = mapped_column(Price, nullable=True)
    volume: Mapped[int | None] = mapped_column(Integer, nullable=True)
    open_interest: Mapped[int | None] = mapped_column(Integer, nullable=True)
    open_interest_change: Mapped[int | None] = mapped_column(Integer, nullable=True)

    implied_volatility: Mapped[float | None] = mapped_column(Float, nullable=True)
    delta: Mapped[float | None] = mapped_column(Float, nullable=True)
    gamma: Mapped[float | None] = mapped_column(Float, nullable=True)
    theta: Mapped[float | None] = mapped_column(Float, nullable=True)
    vega: Mapped[float | None] = mapped_column(Float, nullable=True)

    snapshot: Mapped[MarketSnapshot] = relationship(back_populates="options")


class AnalysisCycle(Base, Recorded):
    """One run of the brain, with its reasoning — spec §31.

    Evidence lists are kept, not just the conclusions. A regime of
    TREND_DOWN at 0.29 confidence is not reviewable; the same classification
    with "20-bar regression slope -27.92 pts/bar" beside it is. Reconstructing
    a decision means reading why, and why does not survive in a label.
    """

    __tablename__ = "analysis_cycles"
    __table_args__ = (Index("ix_cycle_symbol_time", "index_symbol", "observed_at"),)

    snapshot_id: Mapped[uuid.UUID | None] = mapped_column(
        PortableUUID, ForeignKey("market_snapshots.id", ondelete="CASCADE"),
        nullable=True,
    )
    index_symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    regime: Mapped[str] = mapped_column(String(24), nullable=False)
    regime_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    regime_evidence: Mapped[list[str]] = mapped_column(JSON, default=list)

    signal_direction: Mapped[str] = mapped_column(String(12), nullable=False)
    signal_score: Mapped[float] = mapped_column(Float, nullable=False)
    signal_evidence: Mapped[list[str]] = mapped_column(JSON, default=list)

    strategy: Mapped[str] = mapped_column(String(32), nullable=False)
    is_actionable: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_authorized: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    authorization_blocked_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    index_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    constituent_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    options_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    volatility_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    basis_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    candidate: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    """The ranked structure, or NULL when nothing was ranked.

    NULL and `{}` mean different things here — nothing ranked versus a
    structure with no legs — so the column stays nullable rather than
    defaulting to an empty object.
    """

    snapshot: Mapped[MarketSnapshot | None] = relationship(back_populates="cycles")


class TradeFeedbackRow(Base, Recorded):
    """Spec §19. A closed position's outcome against what was expected.

    Recorded, never acted on: nothing in this table may change a live
    parameter. The Learning pipeline reads it and writes `lessons`, and a
    lesson still needs validation and approval before it becomes a strategy
    version.
    """

    __tablename__ = "trade_feedback"
    __table_args__ = (Index("ix_feedback_thesis", "thesis_id"),)

    feedback_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    thesis_id: Mapped[str] = mapped_column(String(64), nullable=False)
    original_thesis: Mapped[str] = mapped_column(Text, nullable=False)
    scenario_id: Mapped[str] = mapped_column(String(64), nullable=False)
    strategy: Mapped[str] = mapped_column(String(32), nullable=False)
    strike_summary: Mapped[str] = mapped_column(Text, nullable=False)
    risk_summary: Mapped[str] = mapped_column(Text, nullable=False)
    expected_behavior: Mapped[str] = mapped_column(Text, nullable=False)
    actual_behavior: Mapped[str] = mapped_column(Text, nullable=False)
    exit_reason: Mapped[str] = mapped_column(String(64), nullable=False)
    pnl: Mapped[Any] = mapped_column(Price, nullable=False)
    thesis_confirmed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    market_conditions: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class LessonRow(Base, Recorded):
    """Spec §20. A pattern drawn from feedback, awaiting validation.

    `validated` is a fact about evidence; `approved_at` is a human act. Both
    are needed before a lesson can inform a strategy version, and neither is
    ever set by the pipeline that derives the lesson.
    """

    __tablename__ = "lessons"

    lesson_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    derived_from_feedback_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    validated: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    supporting_trades: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    approved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    approved_by: Mapped[str | None] = mapped_column(String(128), nullable=True)


class StrategyVersionRow(Base, Recorded):
    """Spec §20. A proposed parameter set, and whether anyone approved it.

    Proposals land here `is_production=False` and stay that way until a
    person promotes them. There is deliberately no code path in this system
    that flips it — see `learning/pipeline.py`.
    """

    __tablename__ = "strategy_versions"

    strategy_version: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False
    )
    """The strategy's own version label.

    Not `version` — that name is taken by Recorded's row-revision counter
    from spec §27, and two different meanings under one attribute is how a
    row's revision ends up written into a strategy identifier.
    """
    parameters: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    derived_from_lesson_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    backtest_summary: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    is_production: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    approved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    approved_by: Mapped[str | None] = mapped_column(String(128), nullable=True)


class SystemEventRow(Base, Recorded):
    """Spec §27 system_events. Operational facts, not market ones.

    Feed outages, capture runs, seed failures, kill-switch trips. Kept in the
    same database as the market data so an odd analysis can be lined up
    against whatever the system was doing at the time.
    """

    __tablename__ = "system_events"
    __table_args__ = (Index("ix_sysevent_kind_time", "kind", "occurred_at"),)

    kind: Mapped[str] = mapped_column(String(48), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), default="info", nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    detail: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
