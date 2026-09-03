"""Spec §16. The last thing between a decision and the broker.

Only the deterministic execution layer may talk to a broker, and only after
every mandatory check below passes. If any mandatory check fails: NO ORDER.
There is no override path — no argument, flag, or agent input can widen a
check, and `validate` takes no parameter that could carry one. The check for
that is a test, not a comment.

Why the gate re-validates instead of trusting the decision
----------------------------------------------------------
A `TradeDecision` is a statement about a market that existed when the
analysis ran. Between that moment and this one the spread can widen, the
strike's bid can disappear, the session can end, the day's loss limit can be
hit by an unrelated position, and the same thesis can already have been sent.
Trusting the decision's own numbers would make the gate a formality. So every
price, spread and liquidity check here reads the **live** chain, and every
limit check reads the **live** portfolio — the decision supplies intent, not
evidence.

That means the gate legitimately rejects decisions that risk approved
seconds earlier. That is the gate working.

Two design choices worth stating
--------------------------------
**Every failing check is reported, not the first.** An operator looking at a
blocked order needs the whole picture: "spread too wide" and "also outside
market hours" lead to different actions than either alone.

**Legs are sequenced so the risk-reducing one goes first.** On a credit
spread the long leg is the protection: submit the short leg first and a
failure on the long leg leaves a naked short position, which is the single
worst outcome available to this system. Indian brokers also grant spread
margin only once the hedge is present, so buying first is both safer and
cheaper. The Order Manager must honour `OrderRequest.sequence`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime, time, timedelta, timezone
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from index_option_brain.contracts.decision import TradeDecision
from index_option_brain.contracts.enums import (
    MarketSessionState,
    OrderSide,
    TradeDecisionType,
)
from index_option_brain.contracts.instruments import (
    AccountSnapshot,
    IndexSpec,
    OptionQuote,
)
from index_option_brain.contracts.order import OrderRequest
from index_option_brain.contracts.risk import PortfolioState
from index_option_brain.contracts.strike import StrikeCandidate, StrikeLeg
from index_option_brain.risk.limits import RiskLimits
from index_option_brain.risk.margin import DefinedRiskMarginModel, MarginModel

IST = timezone(timedelta(hours=5, minutes=30), name="IST")

FailureRecorder = Callable[["ExecutionCheck", str], None]
"""Records one failed check and its reason.

Passed into each check rather than having checks return values, so a check
can report several distinct failures — a leg can be both illiquid and
mis-priced — and so no check can accidentally short-circuit the rest.
"""


class ExecutionCheck(StrEnum):
    """The mandatory checks of spec §16. All sixteen are blocking."""

    DECISION_VALID = "decision_valid"
    RISK_APPROVED = "risk_approved"
    INSTRUMENT_VALID = "instrument_valid"
    EXPIRY_VALID = "expiry_valid"
    STRIKE_VALID = "strike_valid"
    LOT_SIZE_VALID = "lot_size_valid"
    QUANTITY_VALID = "quantity_valid"
    PRICE_VALID = "price_valid"
    LIQUIDITY_VALID = "liquidity_valid"
    SPREAD_ACCEPTABLE = "spread_acceptable"
    MARGIN_AVAILABLE = "margin_available"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    POSITION_LIMIT = "position_limit"
    DUPLICATE_ORDER_CHECK = "duplicate_order_check"
    KILL_SWITCH = "kill_switch"
    MARKET_SESSION = "market_session"


class ExecutionGateConfig(BaseModel):
    """Gate thresholds. Portfolio and margin limits come from `RiskLimits`
    instead of being restated here — two sources of truth for "max open
    positions" is exactly the drift that lets a limit be enforced in one place
    and not the other.
    """

    model_config = ConfigDict(frozen=True)

    entry_cutoff_ist: time = time(15, 0)
    """No new entries after this. The last half hour of an Indian session is
    where spreads widen and closing auctions distort marks, and a position
    opened there has no time to be managed."""
    allow_entry_in_closing: bool = False
    max_price_deviation: float = 0.10
    """How far the live mid may sit from the price the decision was built on,
    as a fraction. Beyond this the structure being priced is not the structure
    that was analysed, so the economics — max loss, breakeven — no longer
    hold."""
    min_open_interest: int = 250
    """In lots, matching what NSE publishes."""
    min_traded_volume: int = 100
    require_two_sided_quote: bool = True
    """A leg with no bid cannot be exited at any price, only abandoned."""
    max_price_ticks_from_mid: int = 20
    """A limit price further than this from the live mid is treated as a
    fat-finger rather than an aggressive order."""
    margin_headroom: Decimal = Decimal("1.10")
    """Estimated margin is multiplied by this before being compared with
    available margin. Margin is SPAN + exposure, computed by the exchange and
    only estimated here, and an order rejected for insufficient margin after
    one leg has filled is the worst way to discover the estimate was tight."""


class ExecutionContext(BaseModel):
    """The live world the gate checks a decision against.

    Passed as one object so a new check cannot be added without the state it
    needs appearing on the contract (spec §3: no uncontrolled variables
    between modules). Notably absent: anything that could relax a check.
    """

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    session_state: MarketSessionState
    index_spec: IndexSpec
    chain: list[OptionQuote]
    """The live chain. Prices, spreads and liquidity are read from here, never
    from the decision."""
    account: AccountSnapshot
    portfolio: PortfolioState
    kill_switch_engaged: bool = False
    pending_thesis_ids: frozenset[str] = Field(default_factory=frozenset)
    """Theses with orders already in flight. Open positions come from
    `portfolio`; this covers the window between submission and fill, which is
    exactly when a duplicate is easiest to send."""

    def quote_for(self, instrument_key: str) -> OptionQuote | None:
        for quote in self.chain:
            if quote.contract.instrument_key == instrument_key:
                return quote
        return None


class ExecutionGateResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    approved: bool
    failed_checks: list[ExecutionCheck] = Field(default_factory=list)
    order_requests: list[OrderRequest] = Field(default_factory=list)
    """Empty unless every check passed. One request per leg, ordered by
    `sequence` with risk-reducing legs first."""
    evidence: list[str] = Field(default_factory=list)
    """Why each failing check failed, in terms an operator can act on."""

    @property
    def passed_all(self) -> bool:
        return self.approved and not self.failed_checks

    @classmethod
    def blocked(
        cls, checks: list[ExecutionCheck], evidence: list[str]
    ) -> ExecutionGateResult:
        """A rejection carries no order requests, ever.

        Constructed through this classmethod so there is no code path that
        builds a blocked result and an order request in the same object — a
        caller that read `order_requests` without checking `approved` would
        otherwise send an order the gate refused.
        """
        return cls(approved=False, failed_checks=checks, order_requests=[], evidence=evidence)


class ExecutionGate(ABC):
    @abstractmethod
    def validate(
        self, decision: TradeDecision, context: ExecutionContext
    ) -> ExecutionGateResult:
        """Run every ExecutionCheck against `decision` and the live `context`.

        Must return approved=False, and no OrderRequest, unless every
        mandatory check passes. There is deliberately no third parameter: an
        override argument here would defeat the purpose of the layer.
        """
        ...


class DeterministicExecutionGate(ExecutionGate):
    """The production gate. No randomness, no network, no clock reads.

    Everything it decides on comes from the decision and the context, which is
    what lets the same gate run in live, paper, backtest and replay
    (spec §22) and produce the same answer for the same inputs.
    """

    def __init__(
        self,
        limits: RiskLimits | None = None,
        config: ExecutionGateConfig | None = None,
        margin_model: MarginModel | None = None,
    ) -> None:
        self._limits = limits or RiskLimits()
        self._config = config or ExecutionGateConfig()
        self._margin = margin_model or DefinedRiskMarginModel()

    def validate(
        self, decision: TradeDecision, context: ExecutionContext
    ) -> ExecutionGateResult:
        try:
            return self._validate(decision, context)
        # Fail closed. A gate that raises leaves the caller with no answer,
        # and the safe answer to "may I send this order" is always no. The
        # exception text is preserved as evidence so the failure is
        # diagnosable rather than merely silent.
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see above
            return ExecutionGateResult.blocked(
                [ExecutionCheck.DECISION_VALID],
                [
                    (
                        f"Execution gate raised {type(exc).__name__}: {exc}. "
                        "No order was produced."
                    )
                ],
            )

    # ------------------------------------------------------------- internals

    def _validate(
        self, decision: TradeDecision, context: ExecutionContext
    ) -> ExecutionGateResult:
        failed: list[ExecutionCheck] = []
        evidence: list[str] = []

        def fail(check: ExecutionCheck, reason: str) -> None:
            if check not in failed:
                failed.append(check)
            evidence.append(reason)

        # Environment first: these do not depend on the decision at all, and
        # when the kill switch is on or the market is shut, nothing else about
        # the order matters.
        self._check_kill_switch(context, fail)
        self._check_session(context, fail)
        self._check_decision(decision, fail)
        self._check_risk_approval(decision, fail)
        self._check_duplicate(decision, context, fail)
        self._check_daily_loss(context, fail)
        self._check_position_limits(decision, context, fail)

        structure = decision.contracts[0] if decision.contracts else None
        # `lots`, never `quantity`: a RiskDecision carries both, and quantity
        # is in units (lots x lot size). Reading the wrong one here would size
        # every order by a factor of the lot size, and the mistake is
        # invisible in the number itself.
        lots = decision.risk_decision.lots

        if structure is not None:
            for leg in structure.legs:
                self._check_leg(leg, decision, context, fail)
            self._check_quantity(structure, lots, context, fail)
            self._check_margin(structure, lots, context, fail)

        if failed:
            return ExecutionGateResult.blocked(failed, evidence)

        assert structure is not None  # guaranteed by DECISION_VALID
        return ExecutionGateResult(
            approved=True,
            failed_checks=[],
            order_requests=self._build_orders(decision, structure, lots, context),
            evidence=[
                (
                    f"All {len(ExecutionCheck)} mandatory checks passed for "
                    f"{decision.strategy} x{lots} lot(s)"
                )
            ],
        )

    # ------------------------------------------------------- environment

    def _check_kill_switch(self, context: ExecutionContext, fail: FailureRecorder) -> None:
        if context.kill_switch_engaged:
            fail(
                ExecutionCheck.KILL_SWITCH,
                "Kill switch is engaged — no orders may be sent while it is on",
            )

    def _check_session(self, context: ExecutionContext, fail: FailureRecorder) -> None:
        state = context.session_state
        local = context.timestamp.astimezone(IST)

        allowed = {MarketSessionState.ACTIVE}
        if self._config.allow_entry_in_closing:
            allowed.add(MarketSessionState.CLOSING)

        if state not in allowed:
            fail(
                ExecutionCheck.MARKET_SESSION,
                f"Session state is {state} — entries are only permitted in "
                f"{sorted(s.value for s in allowed)}",
            )
            return

        if local.time() >= self._config.entry_cutoff_ist:
            fail(
                ExecutionCheck.MARKET_SESSION,
                f"{local:%H:%M} IST is past the "
                f"{self._config.entry_cutoff_ist:%H:%M} entry cutoff — a position "
                "opened now has no session left to be managed in",
            )

    # ---------------------------------------------------------- decision

    def _check_decision(self, decision: TradeDecision, fail: FailureRecorder) -> None:
        if decision.decision is not TradeDecisionType.EXECUTE:
            fail(
                ExecutionCheck.DECISION_VALID,
                f"Decision is {decision.decision}, not EXECUTE — only an EXECUTE "
                "decision may reach a broker",
            )
        if not decision.contracts:
            fail(
                ExecutionCheck.DECISION_VALID,
                "Decision carries no structure to execute",
            )
        elif len(decision.contracts) > 1:
            # More than one structure is an assembly error, not a basket: the
            # risk decision sized exactly one.
            fail(
                ExecutionCheck.DECISION_VALID,
                f"Decision carries {len(decision.contracts)} structures; risk "
                "authorized one",
            )
        elif not decision.contracts[0].legs:
            fail(ExecutionCheck.DECISION_VALID, "Structure has no legs")
        if not decision.thesis_id:
            fail(
                ExecutionCheck.DECISION_VALID,
                "Decision has no thesis_id, so the resulting position could not "
                "be traced back to its reasoning",
            )
        if decision.max_loss <= 0 and decision.contracts:
            fail(
                ExecutionCheck.DECISION_VALID,
                "Decision reports a max loss of zero, which no real structure has",
            )

    def _check_risk_approval(self, decision: TradeDecision, fail: FailureRecorder) -> None:
        risk = decision.risk_decision
        if not risk.approved:
            codes = [str(code) for code in risk.reason_codes]
            fail(
                ExecutionCheck.RISK_APPROVED,
                f"Risk did not approve this trade: {codes or ['no reason given']}",
            )
        if risk.lots <= 0:
            fail(
                ExecutionCheck.RISK_APPROVED,
                f"Risk authorized {risk.lots} lots — there is nothing to send",
            )
        elif risk.quantity != risk.lots * self._structure_lot_size(decision):
            # The two size fields must agree, or one of them is being written
            # in the wrong unit somewhere upstream.
            fail(
                ExecutionCheck.QUANTITY_VALID,
                f"Risk reports {risk.lots} lots but {risk.quantity} units, which "
                "do not correspond at this contract's lot size",
            )

    def _structure_lot_size(self, decision: TradeDecision) -> int:
        """The lot size the risk decision's unit count should be built from."""
        if not decision.contracts or not decision.contracts[0].legs:
            return 0
        return decision.contracts[0].legs[0].contract.lot_size

    # ------------------------------------------------------------- limits

    def _check_duplicate(
        self,
        decision: TradeDecision,
        context: ExecutionContext,
        fail: FailureRecorder,
    ) -> None:
        """One thesis, one position.

        Both an open position and an in-flight order count. Without the
        in-flight half, a cycle that runs again before the first fill arrives
        would send the same trade twice, and the two would look like one
        position of double the size.
        """
        if decision.thesis_id in context.pending_thesis_ids:
            fail(
                ExecutionCheck.DUPLICATE_ORDER_CHECK,
                f"Thesis {decision.thesis_id} already has an order in flight",
            )
        for position in context.portfolio.open_positions:
            if position.thesis_id == decision.thesis_id and position.is_open:
                fail(
                    ExecutionCheck.DUPLICATE_ORDER_CHECK,
                    f"Thesis {decision.thesis_id} is already open as position "
                    f"{position.position_id}",
                )
                break

    def _check_daily_loss(self, context: ExecutionContext, fail: FailureRecorder) -> None:
        equity = context.account.net_equity
        if equity <= 0:
            fail(
                ExecutionCheck.DAILY_LOSS_LIMIT,
                f"Account equity is {equity}, so no loss budget can be computed",
            )
            return
        allowed_loss = equity * self._limits.max_daily_loss
        day_pnl = context.portfolio.day_pnl
        if day_pnl <= -allowed_loss:
            fail(
                ExecutionCheck.DAILY_LOSS_LIMIT,
                f"Day P&L {day_pnl} has reached the daily loss limit of "
                f"{-allowed_loss} ({self._limits.max_daily_loss:.1%} of equity)",
            )

    def _check_position_limits(
        self,
        decision: TradeDecision,
        context: ExecutionContext,
        fail: FailureRecorder,
    ) -> None:
        portfolio = context.portfolio
        limits = self._limits
        if portfolio.open_position_count >= limits.max_open_positions:
            fail(
                ExecutionCheck.POSITION_LIMIT,
                f"{portfolio.open_position_count} positions open, limit is "
                f"{limits.max_open_positions}",
            )
        if portfolio.count_for_strategy(decision.strategy) >= limits.max_positions_per_strategy:
            fail(
                ExecutionCheck.POSITION_LIMIT,
                f"Already at the {limits.max_positions_per_strategy}-position "
                f"limit for {decision.strategy}",
            )
        symbol = decision.underlying_symbol
        if symbol and portfolio.count_for_underlying(symbol) >= limits.max_positions_per_underlying:
            fail(
                ExecutionCheck.POSITION_LIMIT,
                f"Already at the {limits.max_positions_per_underlying}-position "
                f"limit for {symbol}",
            )

    # --------------------------------------------------------------- legs

    def _check_leg(
        self,
        leg: StrikeLeg,
        decision: TradeDecision,
        context: ExecutionContext,
        fail: FailureRecorder,
    ) -> None:
        contract = leg.contract
        key = contract.instrument_key
        spec = context.index_spec

        if decision.underlying_symbol and contract.underlying_symbol != decision.underlying_symbol:
            fail(
                ExecutionCheck.INSTRUMENT_VALID,
                f"Leg {key} is on {contract.underlying_symbol} but the decision is "
                f"on {decision.underlying_symbol}",
            )
        if contract.trading_status != "active":
            fail(
                ExecutionCheck.INSTRUMENT_VALID,
                f"Leg {key} has trading status {contract.trading_status!r}",
            )

        quote = context.quote_for(key)
        if quote is None:
            # Not pedantry: an instrument absent from the live chain cannot be
            # priced, cannot be checked for liquidity, and may not be
            # tradeable at all.
            fail(
                ExecutionCheck.INSTRUMENT_VALID,
                f"Leg {key} is not present in the live chain",
            )

        expiry_date = context.timestamp.astimezone(IST).date()
        if contract.expiry < expiry_date:
            fail(
                ExecutionCheck.EXPIRY_VALID,
                f"Leg {key} expired on {contract.expiry.isoformat()}",
            )

        if contract.strike <= 0:
            fail(ExecutionCheck.STRIKE_VALID, f"Leg {key} has a non-positive strike")
        elif spec.strike_step is None:
            # No uniform step, so listedness cannot be verified this way —
            # some venues widen the ladder away from the money. The check
            # fails rather than passing: this gate is the last thing between
            # a decision and a live order, and "could not verify" must not
            # render as "verified". A venue with an irregular ladder needs
            # the strike checked against the actual instrument list.
            fail(
                ExecutionCheck.STRIKE_VALID,
                f"Leg {key} cannot be verified as a listed strike: "
                f"{spec.symbol} publishes no uniform strike step, so a "
                "multiple-of-step test says nothing",
            )
        elif spec.strike_step > 0 and contract.strike % spec.strike_step != 0:
            fail(
                ExecutionCheck.STRIKE_VALID,
                f"Strike {contract.strike} is not a multiple of the "
                f"{spec.strike_step} strike step, so it is not a listed strike",
            )

        if contract.lot_size != spec.lot_size:
            # The lot size in a contract spec is exchange configuration that
            # gets revised by circular. A stale one silently mis-sizes every
            # order built from it.
            fail(
                ExecutionCheck.LOT_SIZE_VALID,
                f"Leg {key} carries lot size {contract.lot_size} but "
                f"{spec.symbol} trades in lots of {spec.lot_size}",
            )
        if leg.lots <= 0:
            fail(
                ExecutionCheck.QUANTITY_VALID,
                f"Leg {key} has a ratio of {leg.lots} lots",
            )

        if quote is None:
            return

        self._check_leg_price(leg, quote, fail)
        self._check_leg_liquidity(leg, quote, fail)

    def _check_leg_price(
        self, leg: StrikeLeg, quote: OptionQuote, fail: FailureRecorder
    ) -> None:
        key = leg.contract.instrument_key
        live_mid = quote.mid
        if live_mid <= 0:
            fail(ExecutionCheck.PRICE_VALID, f"Leg {key} has no usable live price")
            return

        reference = leg.reference_price
        if reference <= 0:
            fail(
                ExecutionCheck.PRICE_VALID,
                f"Leg {key} was priced at {reference} in the decision",
            )
            return

        drift = abs(live_mid - reference) / reference
        if drift > Decimal(str(self._config.max_price_deviation)):
            fail(
                ExecutionCheck.PRICE_VALID,
                f"Leg {key} was analysed at {reference} and is now {live_mid} "
                f"({drift:.1%} away, limit {self._config.max_price_deviation:.1%}) — "
                "the structure's max loss and breakeven no longer hold",
            )

        tick = leg.contract.tick_size
        if tick > 0 and reference % tick != 0:
            fail(
                ExecutionCheck.PRICE_VALID,
                f"Leg {key} price {reference} is not a multiple of the {tick} tick",
            )

    def _check_leg_liquidity(
        self, leg: StrikeLeg, quote: OptionQuote, fail: FailureRecorder
    ) -> None:
        key = leg.contract.instrument_key
        cfg = self._config

        if cfg.require_two_sided_quote and (quote.bid is None or quote.ask is None):
            fail(
                ExecutionCheck.LIQUIDITY_VALID,
                f"Leg {key} is quoted on one side only — it could be entered but "
                "not exited",
            )
        if quote.open_interest < cfg.min_open_interest:
            fail(
                ExecutionCheck.LIQUIDITY_VALID,
                f"Leg {key} has {quote.open_interest} lots of open interest, "
                f"below the {cfg.min_open_interest} floor",
            )
        if quote.volume < cfg.min_traded_volume:
            fail(
                ExecutionCheck.LIQUIDITY_VALID,
                f"Leg {key} has traded {quote.volume} today, below the "
                f"{cfg.min_traded_volume} floor",
            )

        spread = quote.relative_spread
        if spread is None:
            if cfg.require_two_sided_quote:
                fail(
                    ExecutionCheck.SPREAD_ACCEPTABLE,
                    f"Leg {key} has no measurable spread",
                )
        elif float(spread) > self._limits.max_relative_spread:
            fail(
                ExecutionCheck.SPREAD_ACCEPTABLE,
                f"Leg {key} spread is {float(spread):.2%} of mid, above the "
                f"{self._limits.max_relative_spread:.2%} ceiling — and on a "
                "round trip it is paid twice",
            )

    # --------------------------------------------------------- size, margin

    def _check_quantity(
        self,
        structure: StrikeCandidate,
        lots: int,
        context: ExecutionContext,
        fail: FailureRecorder,
    ) -> None:
        limits = self._limits
        if lots <= 0:
            return  # already reported by RISK_APPROVED
        if lots > limits.max_lots:
            fail(
                ExecutionCheck.QUANTITY_VALID,
                f"{lots} lots exceeds the {limits.max_lots}-lot per-trade cap",
            )
        if lots < limits.min_lots:
            fail(
                ExecutionCheck.QUANTITY_VALID,
                f"{lots} lots is below the {limits.min_lots}-lot minimum",
            )
        for leg in structure.legs:
            units = leg.lots * lots * leg.contract.lot_size
            if units % leg.contract.lot_size != 0:
                fail(
                    ExecutionCheck.QUANTITY_VALID,
                    f"Leg {leg.contract.instrument_key} would be sent as {units} "
                    f"units, which is not a whole number of "
                    f"{leg.contract.lot_size}-unit lots",
                )

    def _check_margin(
        self,
        structure: StrikeCandidate,
        lots: int,
        context: ExecutionContext,
        fail: FailureRecorder,
    ) -> None:
        if lots <= 0:
            return
        required = self._margin.estimate(structure, lots) * self._config.margin_headroom
        available = context.account.available_margin
        if required > available:
            fail(
                ExecutionCheck.MARGIN_AVAILABLE,
                f"Estimated margin {required.quantize(Decimal('0.01'))} (including "
                f"{self._config.margin_headroom}x headroom) exceeds available "
                f"margin {available}",
            )
            return

        utilization_cap = available * self._limits.max_margin_utilization
        if required > utilization_cap:
            fail(
                ExecutionCheck.MARGIN_AVAILABLE,
                f"Estimated margin {required.quantize(Decimal('0.01'))} would use "
                f"more than the permitted "
                f"{self._limits.max_margin_utilization:.0%} of available margin "
                f"({utilization_cap.quantize(Decimal('0.01'))})",
            )

    # ------------------------------------------------------------- orders

    def _build_orders(
        self,
        decision: TradeDecision,
        structure: StrikeCandidate,
        lots: int,
        context: ExecutionContext,
    ) -> list[OrderRequest]:
        """One request per leg, risk-reducing legs first.

        The limit price is the live mid rather than the decision's reference
        price: the reference is what the structure was analysed at, and by now
        it is stale by however long the cycle took. Mid rather than the touch
        because paying the full spread on entry is a cost the structure was
        not priced with — the Order Manager owns any subsequent walking of the
        price, since that is order handling rather than authorization.
        """
        ordered = sorted(
            structure.legs,
            key=lambda leg: (0 if leg.side is OrderSide.BUY else 1, str(leg.contract.strike)),
        )
        requests: list[OrderRequest] = []
        for sequence, leg in enumerate(ordered):
            quote = context.quote_for(leg.contract.instrument_key)
            limit = quote.mid if quote is not None else leg.reference_price
            tick = leg.contract.tick_size
            if tick > 0:
                limit = (limit / tick).quantize(Decimal(1)) * tick
            leg_lots = leg.lots * lots
            requests.append(
                OrderRequest(
                    decision_id=decision.decision_id,
                    thesis_id=decision.thesis_id,
                    contract=leg.contract,
                    side=leg.side,
                    quantity=leg_lots * leg.contract.lot_size,
                    lots=leg_lots,
                    limit_price=limit,
                    sequence=sequence,
                )
            )
        return requests
