"""Concrete persistence for spec §21, behind the `memory` contracts.

Two repositories, split by what the caller is doing rather than by table:

* `SnapshotRepository` records what was *observed* — the market, the chain,
  and each analysis cycle. Write-heavy, runs on every poll, and must never
  raise into the analysis loop.
* `SqlTradeMemoryRepository` records what was *decided and learned* — trade
  feedback, lessons, strategy versions. Low volume, and read back by the
  learning pipeline.

Idempotency
-----------
The capture loop polls on a timer and restarts with the process, so it will
be asked to record the same moment twice. `record_snapshot` keys on
(index_symbol, observed_at) — the feed's own timestamp — and returns the
existing row rather than inserting a second copy. This is why the feed
timestamp is stored separately from `created_at`: dedup on wall clock would
never match, and every restart would double the corpus.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from index_option_brain.brain.pipeline import BrainCycleResult
from index_option_brain.contracts.feedback import Lesson, TradeFeedback
from index_option_brain.contracts.market_state import MarketState
from index_option_brain.database.models import (
    AnalysisCycle,
    LessonRow,
    MarketSnapshot,
    OptionSnapshot,
    StrategyVersionRow,
    SystemEventRow,
    TradeFeedbackRow,
)

logger = logging.getLogger(__name__)


def _decimal(value: Any) -> Decimal | None:
    return None if value is None else Decimal(str(value))


def _naive_utc(moment: datetime | None) -> datetime | None:
    """Normalise to tz-aware UTC.

    SQLite has no timezone type, so a value written aware comes back naive
    and compares unequal to the one that went in. Everything is stored as
    UTC and re-tagged on read; the alternative is a dedup key that silently
    stops matching on the SQLite path.
    """
    if moment is None:
        return None
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


class SnapshotRepository:
    """Records observations. Never raises into the caller's loop."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def find_snapshot(
        self, index_symbol: str, observed_at: datetime
    ) -> MarketSnapshot | None:
        moment = _naive_utc(observed_at)
        result = await self._session.execute(
            select(MarketSnapshot).where(
                MarketSnapshot.index_symbol == index_symbol.upper(),
                MarketSnapshot.observed_at == moment,
            )
        )
        return result.scalar_one_or_none()

    async def record_snapshot(
        self, state: MarketState, *, with_chain: bool = True
    ) -> MarketSnapshot:
        """Persist one market state, and its chain when asked.

        Returns the existing row unchanged if this moment is already held.
        `with_chain` is separate because the chain is by far the largest
        thing written — a NIFTY snapshot is ~170 legs — and a caller polling
        every 20 seconds wants the state every time and the chain far less
        often.
        """
        symbol = state.index_symbol.upper()
        observed = _naive_utc(state.timestamp)
        assert observed is not None

        existing = await self.find_snapshot(symbol, observed)
        if existing is not None:
            return existing

        quote = state.index_state.quote
        options = state.options_state
        volatility = state.volatility_state
        breadth = state.constituent_state.quotes

        row = MarketSnapshot(
            index_symbol=symbol,
            observed_at=observed,
            session_state=str(state.session_state),
            spot=_decimal(quote.ltp),
            day_open=_decimal(quote.open),
            day_high=_decimal(quote.high),
            day_low=_decimal(quote.low),
            previous_close=_decimal(quote.previous_close),
            india_vix=volatility.india_vix,
            atm_iv=volatility.atm_iv,
            expiry=(
                datetime.combine(options.expiry, datetime.min.time(), tzinfo=UTC)
                if options.expiry
                else None
            ),
            forward=_decimal(options.forward),
            forward_excess_basis=_decimal(options.forward_excess_basis),
            forward_strikes_used=options.forward_strikes_used or None,
            # Advances and declines are counted here from the quotes actually
            # observed, not read off the exchange's own tally: the two can
            # disagree (NSE's deadband differs), and the stored number must
            # be the one the brains were given.
            breadth_advances=(
                sum(1 for q in breadth if q.change_pct > 0) if breadth else None
            ),
            breadth_declines=(
                sum(1 for q in breadth if q.change_pct < 0) if breadth else None
            ),
            breadth_observed_at=_naive_utc(breadth[0].timestamp) if breadth else None,
            option_legs=len(options.chain),
        )
        self._session.add(row)
        await self._session.flush()

        if with_chain:
            self._add_chain(row, state)
            await self._session.flush()
        return row

    def _add_chain(self, row: MarketSnapshot, state: MarketState) -> None:
        for quote in state.options_state.chain:
            contract = quote.contract
            greeks = quote.greeks
            self._session.add(
                OptionSnapshot(
                    snapshot_id=row.id,
                    underlying_symbol=contract.underlying_symbol.upper(),
                    expiry=datetime.combine(
                        contract.expiry, datetime.min.time(), tzinfo=UTC
                    ),
                    strike=_decimal(contract.strike),
                    option_type=str(contract.option_type),
                    ltp=_decimal(quote.ltp),
                    bid=_decimal(quote.bid),
                    ask=_decimal(quote.ask),
                    volume=quote.volume,
                    open_interest=quote.open_interest,
                    open_interest_change=quote.open_interest_change,
                    implied_volatility=(
                        float(quote.implied_volatility)
                        if quote.implied_volatility is not None
                        else None
                    ),
                    # Stored as computed, not recomputed later: these were
                    # derived against a parity forward that needs the whole
                    # book, and only the mid survives in this table.
                    delta=float(greeks.delta) if greeks else None,
                    gamma=float(greeks.gamma) if greeks else None,
                    theta=float(greeks.theta) if greeks else None,
                    vega=float(greeks.vega) if greeks else None,
                )
            )

    async def record_cycle(
        self, result: BrainCycleResult, *, snapshot: MarketSnapshot | None = None
    ) -> AnalysisCycle:
        """Persist one brain run with its reasoning intact — spec §31."""
        state = result.state
        analysis = state.analysis
        candidate = result.best_candidate

        row = AnalysisCycle(
            snapshot_id=snapshot.id if snapshot else None,
            index_symbol=state.index_symbol.upper(),
            observed_at=_naive_utc(state.timestamp),
            regime=str(result.regime.regime),
            regime_confidence=result.regime.confidence,
            regime_evidence=list(result.regime.evidence),
            signal_direction=str(result.signal.direction),
            signal_score=result.signal.score,
            signal_evidence=list(result.signal.evidence),
            strategy=str(result.selected_strategy),
            is_actionable=result.is_actionable,
            is_authorized=result.is_authorized,
            authorization_blocked_reason=(
                None
                if result.risk_decision is not None
                else "No broker connected, so the Risk Engine has no account to size against"
            ),
            index_confidence=analysis.index.confidence if analysis else None,
            constituent_confidence=(
                analysis.constituents.confidence if analysis else None
            ),
            options_confidence=analysis.options.confidence if analysis else None,
            volatility_confidence=analysis.volatility.confidence if analysis else None,
            basis_score=analysis.options.basis_score if analysis else None,
            vrp_score=analysis.volatility.vrp_score if analysis else None,
            candidate=(
                {
                    "strategy": str(candidate.strategy),
                    "score": candidate.score,
                    "net_premium": float(candidate.net_premium),
                    "max_loss_per_lot": float(candidate.max_loss),
                    "legs": [
                        {
                            "strike": float(leg.contract.strike),
                            "option_type": str(leg.contract.option_type),
                            "side": str(leg.side),
                            "delta": float(leg.delta) if leg.delta is not None else None,
                        }
                        for leg in candidate.legs
                    ],
                }
                if candidate is not None
                else None
            ),
        )
        self._session.add(row)
        await self._session.flush()
        return row

    async def recent_cycles(
        self, index_symbol: str, *, limit: int = 50
    ) -> list[AnalysisCycle]:
        """Newest first — what the console's history panel reads."""
        result = await self._session.execute(
            select(AnalysisCycle)
            .where(AnalysisCycle.index_symbol == index_symbol.upper())
            .order_by(AnalysisCycle.observed_at.desc())
            .limit(limit)
        )
        return list(result.scalars())

    async def chain_coverage(
        self, underlying_symbol: str
    ) -> dict[str, Any]:
        """How much backtest corpus exists, so it can be reported honestly.

        The console shows this because "we are building a corpus" and "we
        have three days of one expiry" look identical without it, and only
        one of them supports a backtest.
        """
        rows = await self._session.execute(
            select(
                MarketSnapshot.observed_at, MarketSnapshot.option_legs
            ).where(
                MarketSnapshot.index_symbol == underlying_symbol.upper(),
                MarketSnapshot.option_legs > 0,
            )
        )
        observed = list(rows.all())
        if not observed:
            return {"snapshots": 0, "sessions": 0, "legs": 0, "first": None, "last": None}
        days = {moment.date() for moment, _ in observed}
        return {
            "snapshots": len(observed),
            "sessions": len(days),
            "legs": sum(legs for _, legs in observed),
            "first": min(days).isoformat(),
            "last": max(days).isoformat(),
        }

    async def record_event(
        self,
        kind: str,
        message: str,
        *,
        severity: str = "info",
        detail: dict[str, Any] | None = None,
        occurred_at: datetime | None = None,
    ) -> None:
        self._session.add(
            SystemEventRow(
                kind=kind,
                severity=severity,
                message=message,
                detail=detail or {},
                occurred_at=_naive_utc(occurred_at) or datetime.now(UTC),
            )
        )

    async def prune_chains(self, *, keep_days: int = 400) -> int:
        """Drop option legs older than `keep_days`, keeping their snapshots.

        The chain is the only table that grows without bound — ~170 legs per
        capture. The market state and the analysis cycle beside it are small
        and stay, so history of what the system saw and decided survives
        even where the full book has been let go.
        """
        cutoff = datetime.now(UTC) - timedelta(days=keep_days)
        stale = await self._session.execute(
            select(OptionSnapshot).where(OptionSnapshot.expiry < cutoff)
        )
        rows = list(stale.scalars())
        for row in rows:
            await self._session.delete(row)
        return len(rows)


class SqlTradeMemoryRepository:
    """Feedback, lessons and strategy versions — the learning record.

    Implements the `TradeMemoryRepository` contract's feedback and lesson
    halves. `get_position_history` is not implemented here: positions have no
    table yet, because the broker's response mapping is unverified and their
    columns would be a guess about a payload nobody has seen. It raises
    rather than returning an empty list, so a caller cannot read "no
    positions" out of "not built".
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save_feedback(self, feedback: TradeFeedback) -> None:
        self._session.add(
            TradeFeedbackRow(
                feedback_id=feedback.feedback_id,
                thesis_id=feedback.thesis_id,
                original_thesis=feedback.original_thesis,
                scenario_id=feedback.scenario_id,
                strategy=feedback.strategy,
                strike_summary=feedback.strike_summary,
                risk_summary=feedback.risk_summary,
                expected_behavior=feedback.expected_behavior,
                actual_behavior=feedback.actual_behavior,
                exit_reason=feedback.exit_reason,
                pnl=feedback.pnl,
                thesis_confirmed=feedback.thesis_confirmed,
                failure_reason=feedback.failure_reason,
                market_conditions=dict(feedback.market_conditions),
                recorded_at=_naive_utc(feedback.recorded_at),
            )
        )
        await self._session.flush()

    async def save_lesson(self, lesson: Lesson) -> None:
        self._session.add(
            LessonRow(
                lesson_id=lesson.lesson_id,
                summary=lesson.summary,
                derived_from_feedback_ids=list(lesson.derived_from_feedback_ids),
                # A lesson arriving from the pipeline is never validated, and
                # never approved. Both are set by a separate act.
                validated=False,
                supporting_trades=len(lesson.derived_from_feedback_ids),
            )
        )
        await self._session.flush()

    async def get_trade_history(self, limit: int = 20) -> list[TradeFeedbackRow]:
        result = await self._session.execute(
            select(TradeFeedbackRow)
            .order_by(TradeFeedbackRow.recorded_at.desc())
            .limit(limit)
        )
        return list(result.scalars())

    async def get_lessons(self, *, validated_only: bool = False) -> list[LessonRow]:
        query = select(LessonRow).order_by(LessonRow.created_at.desc())
        if validated_only:
            query = query.where(LessonRow.validated.is_(True))
        result = await self._session.execute(query)
        return list(result.scalars())

    async def propose_strategy_version(
        self,
        *,
        parameters: dict[str, Any],
        lesson_ids: list[str],
        backtest_summary: dict[str, Any] | None = None,
    ) -> StrategyVersionRow:
        """Store a proposal. It is never production and this cannot make it so.

        There is no `promote` method on this class on purpose. Flipping
        `is_production` requires a deliberate write from outside the
        pipeline, which is the structural half of spec §20's "no direct
        automatic production parameter mutation".
        """
        row = StrategyVersionRow(
            strategy_version=f"v{datetime.now(UTC):%Y%m%d%H%M%S}-{uuid.uuid4().hex[:6]}",
            parameters=parameters,
            derived_from_lesson_ids=lesson_ids,
            backtest_summary=backtest_summary,
            is_production=False,
        )
        self._session.add(row)
        await self._session.flush()
        return row

    async def production_version(self) -> StrategyVersionRow | None:
        result = await self._session.execute(
            select(StrategyVersionRow)
            .where(StrategyVersionRow.is_production.is_(True))
            .order_by(StrategyVersionRow.approved_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_position_history(self, thesis_id: str) -> list[Any]:
        raise NotImplementedError(
            "Positions have no table yet: the broker adapter's response "
            "mapping is unverified, so their columns would be a guess. "
            "Raising rather than returning [] so this cannot be read as "
            "'no positions'."
        )
