"""Spec §14. Risk has absolute authority over trade authorization.

Three properties define this module, and each one is a deliberate constraint
on how the rest of the system may be written:

1. **It fails closed.** Any exception during evaluation returns a rejection
   carrying `EVALUATION_FAILED`, because spec §29 says a risk-engine failure
   means NO_TRADE. An unevaluable check is a failed check — never a skipped
   one.
2. **It sizes, it does not merely veto.** A candidate arrives priced for one
   lot; this engine decides how many the account can carry, and reports which
   constraint bound that number. Approval without a size would leave the
   sizing decision to whoever called it.
3. **It cannot be overridden.** There is no `force` parameter, no confidence
   input that raises a limit, and no path by which an `AgentAssessment`
   reaches this code. The agent layer imports nothing from here.

Rejections collect *every* applicable reason rather than short-circuiting on
the first. When a trade is refused at 09:20 the useful question is what all
was wrong, not which check happened to run first.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import timedelta
from decimal import Decimal

from index_option_brain.contracts.instruments import AccountSnapshot
from index_option_brain.contracts.risk import (
    PortfolioState,
    RiskDecision,
    RiskReasonCode,
    ScheduledEvent,
    TradeCandidate,
)
from index_option_brain.risk.limits import RiskLimits
from index_option_brain.risk.margin import DefinedRiskMarginModel, MarginModel


class RiskEngine(ABC):
    @abstractmethod
    def authorize(
        self,
        trade: TradeCandidate,
        account: AccountSnapshot,
        portfolio: PortfolioState,
    ) -> RiskDecision: ...


class DeterministicRiskEngine(RiskEngine):
    def __init__(
        self,
        limits: RiskLimits | None = None,
        margin_model: MarginModel | None = None,
    ) -> None:
        self._limits = limits or RiskLimits()
        self._margin = margin_model or DefinedRiskMarginModel()

    @property
    def limits(self) -> RiskLimits:
        return self._limits

    def authorize(
        self,
        trade: TradeCandidate,
        account: AccountSnapshot,
        portfolio: PortfolioState,
    ) -> RiskDecision:
        try:
            return self._authorize(trade, account, portfolio)
        except Exception as exc:  # noqa: BLE001 — fail closed, deliberately broad
            # Spec §29: if the risk engine fails, the answer is NO_TRADE. A
            # bug in a limit check must never read as an approval.
            return RiskDecision.reject(
                RiskReasonCode.EVALUATION_FAILED,
                evidence=[f"Risk evaluation raised {type(exc).__name__}: {exc}"],
            )

    # ------------------------------------------------------------------ core

    def _authorize(
        self,
        trade: TradeCandidate,
        account: AccountSnapshot,
        portfolio: PortfolioState,
    ) -> RiskDecision:
        limits = self._limits
        structure = trade.structure
        equity = account.net_equity

        if equity <= 0:
            return RiskDecision.reject(
                RiskReasonCode.INSUFFICIENT_RISK_BUDGET,
                evidence=[f"Net equity is {equity}; there is nothing to risk"],
            )

        blocking, evidence = self._blocking_checks(trade, account, portfolio, equity)
        if blocking:
            return RiskDecision.reject(*blocking, evidence=evidence)

        sizing = self._size(trade, account, portfolio, equity)
        if sizing.lots < limits.min_lots:
            # A size of zero and a size below a deliberate minimum are
            # different failures: the first means nothing fits, the second
            # means what fits is smaller than is worth transacting.
            codes = list(sizing.binding_codes)
            reasons = [*evidence, *sizing.evidence]
            if sizing.lots > 0:
                codes.insert(0, RiskReasonCode.BELOW_MINIMUM_SIZE)
                reasons.append(
                    f"Affordable size {sizing.lots} lot(s) is below the "
                    f"{limits.min_lots}-lot minimum"
                )
            return RiskDecision.reject(*codes, evidence=reasons)

        lots = sizing.lots
        per_lot_max_loss = structure.max_loss
        max_loss = per_lot_max_loss * lots
        margin = self._margin.estimate(structure, lots)
        lot_size = structure.legs[0].contract.lot_size if structure.legs else 0

        return RiskDecision(
            approved=True,
            reason_codes=[RiskReasonCode.APPROVED],
            max_loss=max_loss,
            quantity=lots * lot_size,
            lots=lots,
            # For defined-risk structures the committed max loss *is* the
            # exposure: there is no scenario in which more can be lost.
            exposure=max_loss,
            margin_required=margin,
            evidence=[
                *evidence,
                *sizing.evidence,
                f"Authorized {lots} lot(s) — {lots * lot_size} contracts",
                (
                    f"Max loss {max_loss} against equity {equity} "
                    f"({float(max_loss / equity) * 100:.2f}%)"
                ),
                f"Estimated margin {margin} of {account.available_margin} available",
            ],
        )

    # -------------------------------------------------------------- checks

    def _blocking_checks(
        self,
        trade: TradeCandidate,
        account: AccountSnapshot,
        portfolio: PortfolioState,
        equity: Decimal,
    ) -> tuple[list[RiskReasonCode], list[str]]:
        """Checks that no position size can satisfy. Every failure is
        collected, not short-circuited."""
        limits = self._limits
        structure = trade.structure
        codes: list[RiskReasonCode] = []
        evidence: list[str] = []

        # --- Structure: is the loss even bounded?
        if structure.max_loss <= 0:
            codes.append(RiskReasonCode.UNDEFINED_RISK_STRUCTURE)
            evidence.append(
                f"Structure reports a max loss of {structure.max_loss}, which cannot be sized"
            )
        elif structure.max_profit is None and not limits.allow_undefined_risk:
            # Long premium has unbounded *profit* and bounded loss, which is
            # fine. This branch exists so the distinction is explicit rather
            # than accidental.
            evidence.append(
                "Unbounded profit with bounded loss — sized on the bounded side"
            )

        if limits.max_loss_ceiling is not None and structure.max_loss > limits.max_loss_ceiling:
            codes.append(RiskReasonCode.MAX_LOSS_ABOVE_CEILING)
            evidence.append(
                f"One lot risks {structure.max_loss}, above the absolute ceiling "
                f"of {limits.max_loss_ceiling}"
            )

        # --- Market quality. The spread is the slippage control: it is paid
        # on the way in and again on the way out.
        if structure.liquidity_score < limits.min_liquidity_score:
            codes.append(RiskReasonCode.LIQUIDITY_BELOW_FLOOR)
            evidence.append(
                f"Liquidity {structure.liquidity_score:.2f} below the "
                f"{limits.min_liquidity_score:.2f} floor"
            )

        if structure.worst_relative_spread > limits.max_relative_spread:
            codes.append(RiskReasonCode.SLIPPAGE_ABOVE_CEILING)
            evidence.append(
                f"Worst leg spread {structure.worst_relative_spread * 100:.2f}% above the "
                f"{limits.max_relative_spread * 100:.2f}% ceiling"
            )

        # --- Daily loss
        daily_budget = equity * limits.max_daily_loss
        if portfolio.day_pnl <= -daily_budget:
            codes.append(RiskReasonCode.DAILY_LOSS_LIMIT_REACHED)
            evidence.append(
                f"Day P&L {portfolio.day_pnl} has reached the daily loss limit "
                f"of {-daily_budget}"
            )

        # --- Position counts
        if portfolio.open_position_count >= limits.max_open_positions:
            codes.append(RiskReasonCode.MAX_POSITIONS_REACHED)
            evidence.append(
                f"{portfolio.open_position_count} open positions, limit "
                f"{limits.max_open_positions}"
            )

        strategy_count = portfolio.count_for_strategy(trade.strategy)
        if strategy_count >= limits.max_positions_per_strategy:
            codes.append(RiskReasonCode.STRATEGY_LIMIT_REACHED)
            evidence.append(
                f"{strategy_count} open {trade.strategy.value} positions, limit "
                f"{limits.max_positions_per_strategy}"
            )

        underlying_count = portfolio.count_for_underlying(trade.underlying_symbol)
        if underlying_count >= limits.max_positions_per_underlying:
            codes.append(RiskReasonCode.INSTRUMENT_LIMIT_REACHED)
            evidence.append(
                f"{underlying_count} open positions on {trade.underlying_symbol}, limit "
                f"{limits.max_positions_per_underlying}"
            )

        # --- Event risk
        event = self._blocking_event(trade, portfolio)
        if event is not None:
            codes.append(RiskReasonCode.EVENT_RISK_BLACKOUT)
            evidence.append(
                f"{event.name} at {event.starts_at:%d %b %H:%M} is inside the "
                f"{limits.event_blackout_hours:.0f}h entry blackout"
            )

        return codes, evidence

    def _blocking_event(
        self, trade: TradeCandidate, portfolio: PortfolioState
    ) -> ScheduledEvent | None:
        """The nearest blocking scheduled event inside the blackout window.

        Timing is measured from the candidate's own state timestamp where the
        portfolio provides one, so a replayed decision blacks out against the
        events of *that* day rather than today's.
        """
        reference = portfolio.account.timestamp
        window = timedelta(hours=self._limits.event_blackout_hours)
        upcoming = [
            e
            for e in portfolio.scheduled_events
            if e.blocks_new_entries and reference <= e.starts_at <= reference + window
        ]
        if not upcoming:
            return None
        return min(upcoming, key=lambda e: e.starts_at)

    # -------------------------------------------------------------- sizing

    class _Sizing:
        def __init__(self) -> None:
            self.lots = 0
            self.binding_codes: list[RiskReasonCode] = []
            self.evidence: list[str] = []

    def _size(
        self,
        trade: TradeCandidate,
        account: AccountSnapshot,
        portfolio: PortfolioState,
        equity: Decimal,
    ) -> DeterministicRiskEngine._Sizing:
        """How many lots each constraint permits; the smallest wins.

        Reporting *which* constraint bound the size is the point — "2 lots,
        limited by available margin" is actionable in a way that "2 lots" is
        not.
        """
        limits = self._limits
        structure = trade.structure
        per_lot_loss = structure.max_loss
        result = self._Sizing()

        # Per-trade risk budget, further limited by what is left of today's
        # loss allowance: one more trade must not be able to breach it.
        trade_budget = equity * limits.max_risk_per_trade
        if limits.max_loss_ceiling is not None:
            trade_budget = min(trade_budget, limits.max_loss_ceiling)

        daily_allowance = equity * limits.max_daily_loss
        daily_remaining = max(Decimal(0), daily_allowance + portfolio.day_pnl)
        risk_budget = min(trade_budget, daily_remaining)

        exposure_budget = max(
            Decimal(0),
            equity * limits.max_portfolio_exposure - portfolio.total_exposure,
        )
        concentration_budget = max(
            Decimal(0),
            equity * limits.max_concentration_per_underlying
            - portfolio.exposure_for_underlying(trade.underlying_symbol),
        )
        margin_budget = max(
            Decimal(0), account.available_margin * limits.max_margin_utilization
        )

        per_lot_margin = self._margin.estimate(structure, 1)

        constraints: list[tuple[int, RiskReasonCode, str]] = [
            (
                self._lots_for(risk_budget, per_lot_loss),
                RiskReasonCode.INSUFFICIENT_RISK_BUDGET,
                f"risk budget {risk_budget} at {per_lot_loss}/lot",
            ),
            (
                self._lots_for(exposure_budget, per_lot_loss),
                RiskReasonCode.EXPOSURE_LIMIT_REACHED,
                f"remaining portfolio exposure {exposure_budget}",
            ),
            (
                self._lots_for(concentration_budget, per_lot_loss),
                RiskReasonCode.CONCENTRATION_LIMIT_REACHED,
                f"remaining {trade.underlying_symbol} concentration {concentration_budget}",
            ),
            (
                self._lots_for(margin_budget, per_lot_margin),
                RiskReasonCode.INSUFFICIENT_MARGIN,
                f"usable margin {margin_budget} at {per_lot_margin}/lot",
            ),
        ]

        allowed = min(count for count, _, _ in constraints)
        capped = min(allowed, limits.max_lots)
        result.lots = max(0, capped)

        binding = [(code, why) for count, code, why in constraints if count == allowed]
        result.binding_codes = [code for code, _ in binding]

        if result.lots < limits.min_lots:
            result.evidence.append(
                "Cannot size to the minimum "
                f"{limits.min_lots} lot(s): " + "; ".join(why for _, why in binding)
            )
        else:
            reason = (
                "the maximum lot cap"
                if capped < allowed
                else "; ".join(why for _, why in binding)
            )
            result.evidence.append(f"Size limited by {reason}")

        return result

    @staticmethod
    def _lots_for(budget: Decimal, per_lot_cost: Decimal) -> int:
        """Whole lots affordable within a budget. A zero cost is treated as
        unconstrained rather than as infinite lots by accident."""
        if per_lot_cost <= 0:
            return 1_000_000
        if budget <= 0:
            return 0
        return int(budget // per_lot_cost)
