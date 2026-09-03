"""FastAPI application: the API the operations console reads.

Every endpoint returns measured data or an explicit unavailable state. There
is no demo mode, no sample payload, and no default that stands in for a
reading the system does not have — the console is built to render "not
connected" as a first-class state, so an endpoint never has to invent
something to keep a panel from looking empty.

Unavailability is a normal response, not an error
-------------------------------------------------
Market and analysis endpoints answer with an envelope carrying
`available: false` and the reason rather than a 5xx. A blocked feed is
information the operator needs displayed in place, and burying it in an error
page makes the one screen that should always be readable the one that breaks.
Genuine server faults still surface as 500s.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from index_option_brain.agent import NarrativeProvider
from index_option_brain.app.live import FeedUnavailable, LiveEngine, session_label
from index_option_brain.app.runner import MarketPoller, PollerConfig
from index_option_brain.capture import CaptureConfig, CaptureRecorder
from index_option_brain.config.settings import Settings, get_settings
from index_option_brain.contracts.provider import Capability, ProviderDescriptor
from index_option_brain.data.bar_store import BarStore
from index_option_brain.data.providers import (
    ALL_PROVIDERS,
    REQUIRED_FOR_ANALYSIS,
    REQUIRED_FOR_TRADING,
    implemented_providers,
    missing_capabilities,
    verified_providers,
)
from index_option_brain.database.engine import Database


def _console_path() -> Path:
    """Where the console lives, in every install mode.

    It ships **inside the package** rather than beside it. The earlier
    repo-relative path worked in a checkout and broke the moment the package
    was pip-installed — `parents[2]` then lands in site-packages, so the
    container built by the Dockerfile served a 404 for its own front page.
    That is the deployment this project recommends, so it was the one place
    the path had to be right.

    `CONSOLE_HTML` overrides it, for serving a modified copy without
    rebuilding the image.
    """
    override = os.environ.get("CONSOLE_HTML")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent / "static" / "console.html"


CONSOLE_HTML = _console_path()


def _capability_names(capabilities: frozenset[Capability]) -> list[str]:
    return sorted(str(capability) for capability in capabilities)


def _describe(provider: ProviderDescriptor, health: Any) -> dict[str, Any]:
    return {
        "provider_id": provider.provider_id,
        "display_name": provider.display_name,
        "kind": str(provider.kind),
        "auth": str(provider.auth),
        "implemented": provider.implemented,
        # Three different questions: code exists, mapping proven against a
        # real payload, and calls that succeeded just now.
        "verified": provider.verified,
        "docs_url": provider.docs_url,
        "notes": list(provider.notes),
        "can_trade": provider.can_trade,
        "capabilities": {
            "all": _capability_names(provider.capabilities),
            "data": _capability_names(provider.data_capabilities),
            "trading": _capability_names(provider.trading_capabilities),
        },
        "credential_fields": [
            {
                "name": field.name,
                "label": field.label,
                "secret": field.secret,
                "required": field.required,
                "help": field.help,
            }
            for field in provider.credential_fields
        ],
        "health": {
            "state": str(health.state),
            "checked_at": health.checked_at,
            "latency_ms": health.latency_ms,
            "last_success_at": health.last_success_at,
            "last_error": health.last_error,
            # Declared is a claim; verified is what a call actually returned.
            "verified_capabilities": _capability_names(health.verified_capabilities),
            "usable": health.is_usable,
        },
    }


def _capture_from(settings: Settings) -> CaptureRecorder | None:
    """The capture recorder configured for this process, or None.

    Defaults to SQLite rather than to nothing, because the chain corpus is
    the only thing here that cannot be recovered later — a box that records
    nothing until someone installs Postgres spends its first weeks throwing
    away the irreplaceable part.
    """
    if not settings.capture_enabled:
        return None
    database = (
        Database(url=settings.database_url)
        if settings.database_url
        else Database.sqlite(settings.sqlite_path)
    )
    return CaptureRecorder(
        database=database,
        config=CaptureConfig(
            chain_interval=timedelta(seconds=settings.capture_chain_seconds)
        ),
    )


def create_app(
    engine: LiveEngine | None = None,
    *,
    poller: MarketPoller | None = None,
    run_poller: bool = True,
) -> FastAPI:
    """Build the app.

    `run_poller=False` is for tests: they drive cycles explicitly rather than
    racing a background loop, and a loop reaching the network from a test
    suite is a test that fails when the market is shut.
    """
    settings = get_settings()
    live = engine or LiveEngine(
        bar_store=BarStore(settings.bar_store_dir) if settings.bar_store_dir else None,
        capture=_capture_from(settings),
    )
    market_poller = poller or MarketPoller(
        live, symbols=("NIFTY", "BANKNIFTY"), config=PollerConfig()
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # The loop is what makes the engine run continuously rather than only
        # when the console is open — and it is the only thing that
        # accumulates bars, since NSE serves no history.
        # Contract specifications first: lot size feeds every sizing
        # calculation, and starting the loop before they are loaded would
        # accumulate a session of state against a fallback table.
        if run_poller:
            await live.ensure_ready()
            await market_poller.start()
        try:
            yield
        finally:
            await market_poller.stop()
            await live.aclose()

    app = FastAPI(title="Index Option Brain", version="0.1.0", lifespan=lifespan)
    app.state.live = live
    app.state.poller = market_poller

    @app.get("/health")
    def health() -> dict[str, object]:
        """Liveness. Always 200 while the process is up.

        Deliberately not gated on the feed: a process manager restarting the
        app because NSE is rate-limiting would turn a data outage into an
        availability outage, and lose the accumulated bars with it.
        """
        return {
            "status": "ok",
            "llm_enabled": settings.llm_enabled,
            "run_mode": settings.run_mode,
        }

    @app.get("/ready")
    def ready() -> JSONResponse:
        """Readiness: is the loop running and getting data.

        This is what an uptime check should watch. It answers 503 when the
        loop has failed its last three polls, because a process that is alive
        but blind looks identical to a healthy one from `/health`.
        """
        snapshot = market_poller.snapshot()
        status = 200 if snapshot["running"] and snapshot["healthy"] else 503
        return JSONResponse(snapshot, status_code=status)

    @app.get("/api/runner")
    def runner() -> dict[str, object]:
        """What the loop has been doing. Counted, not estimated.

        The console shows these so an operator can tell "running and quiet"
        from "running and broken" — which look the same in a single snapshot.
        """
        return market_poller.snapshot()

    @app.get("/api/events")
    def events(symbol: str = "NIFTY", limit: int = 40) -> dict[str, object]:
        """Recently detected triggers, newest first.

        A trigger only ever means "something changed; analyze it" (spec §4),
        so nothing here is an instruction and no payload carries an order.
        """
        recent = market_poller.recent_events(limit=limit)
        return {
            "symbol": symbol.upper(),
            "events": [
                {
                    "event_id": event.event_id,
                    "trigger_type": str(event.trigger_type),
                    "timestamp": event.timestamp.isoformat(),
                    "significance_score": event.significance_score,
                    "payload": event.payload,
                }
                for event in recent
            ],
            "detected": market_poller.stats.events_detected,
            "significant": market_poller.stats.events_significant,
        }

    @app.get("/api/status")
    def status() -> dict[str, object]:
        """What the system is configured to do, and what it cannot do yet."""
        connected = implemented_providers()
        return {
            "run_mode": str(settings.run_mode),
            "llm_enabled": settings.llm_enabled,
            "kill_switch_engaged": settings.kill_switch_enabled,
            "instrument_source": live.instrument_source,
            "trading_enabled": False,
            "trading_blocked_reason": (
                "A broker adapter exists (Dhan), but this process is not wired "
                "to it: no credentials are configured, and its response mapping "
                "has not been verified against live payloads — run "
                "scripts/dhan_probe.py. The system is analysis-only until both "
                "are done."
            ),
            "coverage": {
                "analysis": {
                    "required": _capability_names(REQUIRED_FOR_ANALYSIS),
                    "missing": _capability_names(
                        missing_capabilities(*connected, required=REQUIRED_FOR_ANALYSIS)
                    ),
                },
                "trading": {
                    "required": _capability_names(REQUIRED_FOR_TRADING),
                    "missing": _capability_names(
                        missing_capabilities(*connected, required=REQUIRED_FOR_TRADING)
                    ),
                },
            },
        }

    @app.get("/api/providers")
    async def providers(probe: bool = False, symbol: str = "NIFTY") -> dict[str, Any]:
        """The registry, with health.

        `probe=true` calls the live endpoints and times them. Without it the
        health shown is whatever was last measured, which for a provider never
        called is explicitly NOT_CONFIGURED rather than a zeroed reading.
        """
        if probe:
            try:
                await live.probe(symbol)
            except (FeedUnavailable, OSError):
                # The probe's own failure is recorded in health; the listing
                # must still render.
                pass
        return {
            "providers": [
                _describe(provider, live.health(provider.provider_id))
                for provider in ALL_PROVIDERS
            ],
            "implemented_count": len(implemented_providers()),
            "verified_count": len(verified_providers()),
            "total_count": len(ALL_PROVIDERS),
        }

    @app.get("/api/market/{symbol}")
    async def market(symbol: str) -> dict[str, Any]:
        symbol = symbol.upper()
        try:
            state = await live.market_state(symbol)
        except FeedUnavailable as exc:
            return {"available": False, "symbol": symbol, "reason": str(exc)}

        quote = state.index_state.quote
        options = state.options_state
        volatility = state.volatility_state
        chain = options.chain
        with_greeks = [q for q in chain if q.greeks is not None]

        return {
            "available": True,
            "symbol": symbol,
            "as_of": quote.timestamp.isoformat(),
            "session": {
                "state": str(state.session_state),
                "label": session_label(state.session_state),
            },
            "index": {
                "ltp": float(quote.ltp),
                "open": float(quote.open),
                "high": float(quote.high),
                "low": float(quote.low),
                "previous_close": float(quote.previous_close),
                "change_pct": float(quote.change_pct),
                # None where the provider publishes nothing, never a zero.
                "vwap": float(quote.vwap) if quote.vwap is not None else None,
            },
            "volatility": {
                "india_vix": volatility.india_vix,
                "india_vix_previous_close": volatility.india_vix_previous_close,
                "atm_iv": volatility.atm_iv,
                "realized_volatility": volatility.realized_volatility,
                "days_to_expiry": volatility.days_to_expiry,
                "iv_observations": len(volatility.atm_iv_history),
            },
            "options": {
                "expiry": options.expiry.isoformat() if options.expiry else None,
                "expiry_weekday": (
                    options.expiry.strftime("%A") if options.expiry else None
                ),
                "available_expiries": [
                    expiry.isoformat() for expiry in options.available_expiries[:8]
                ],
                "legs": len(chain),
                "strikes": len({q.contract.strike for q in chain}),
                "legs_with_greeks": len(with_greeks),
                "legs_unmarkable": len(chain) - len(with_greeks),
            },
            "bars": live.bar_coverage(symbol),
            "capture": await live.capture_status(symbol),
            "breadth": {
                "constituents": len(state.constituent_state.quotes),
                "available": bool(state.constituent_state.quotes),
                "as_of": (
                    state.constituent_state.quotes[0].timestamp.isoformat()
                    if state.constituent_state.quotes
                    else None
                ),
                "reason": (
                    None
                    if state.constituent_state.quotes
                    else (
                        "The pre-open auction board is the only constituent feed "
                        "NSE serves; it is stale outside the opening window"
                    )
                ),
            },
            "forward": {
                "value": (
                    float(options.forward) if options.forward is not None else None
                ),
                "basis": (
                    float(options.forward_basis)
                    if options.forward_basis is not None
                    else None
                ),
                "excess_basis": (
                    float(options.forward_excess_basis)
                    if options.forward_excess_basis is not None
                    else None
                ),
                "parity_strikes": options.forward_strikes_used,
                "reason": (
                    None
                    if options.forward is not None
                    else "No strike had a two-sided book on both legs"
                ),
            },
        }

    @app.get("/api/analysis/{symbol}")
    async def analysis(symbol: str) -> dict[str, Any]:
        symbol = symbol.upper()
        try:
            result = await live.analysis(symbol)
        except FeedUnavailable as exc:
            return {"available": False, "symbol": symbol, "reason": str(exc)}

        regime = result.regime
        signal = result.signal
        candidate = result.best_candidate
        # Deterministic, free, and available with LLM_ENABLED=false — which is
        # why the console can render an explanation on every cycle.
        brief = NarrativeProvider().describe(
            analysis=result.state.analysis,
            regime=regime,
            signal=signal,
            strategy=result.selected_strategy,
            candidate=candidate,
            is_authorized=result.is_authorized,
        )

        return {
            "available": True,
            "symbol": symbol,
            "as_of": result.state.timestamp.isoformat(),
            "regime": {
                "type": str(regime.regime) if regime else None,
                "confidence": regime.confidence if regime else None,
                "evidence": list(regime.evidence) if regime else [],
                "scores": dict(regime.scores) if regime else {},
            },
            "signal": {
                "direction": str(signal.direction),
                "score": signal.score,
                "evidence": list(signal.evidence),
            },
            "brief": {
                "summary": brief.summary,
                "supporting": brief.supporting_points,
                "contradicting": brief.contradicting_points,
                "unknowns": brief.unknowns,
                "sources": brief.sources,
                "provider": brief.provider,
            },
            "brains": {
                "index": {
                    "direction": str(result.analysis.index.direction),
                    "confidence": result.analysis.index.confidence,
                    "support": [float(x) for x in result.analysis.index.support_levels],
                    "resistance": [
                        float(x) for x in result.analysis.index.resistance_levels
                    ],
                    "evidence": list(result.analysis.index.evidence),
                },
                "constituents": {
                    "advances": result.analysis.constituents.advances,
                    "declines": result.analysis.constituents.declines,
                    "unchanged": result.analysis.constituents.unchanged,
                    "breadth_score": result.analysis.constituents.breadth_score,
                    "weighted_change_pct": (
                        result.analysis.constituents.weighted_change_pct
                    ),
                    "coverage": result.analysis.constituents.weight_coverage,
                    "confidence": result.analysis.constituents.confidence,
                    "leaders": list(result.analysis.constituents.top_contributors[:5]),
                    "laggards": list(result.analysis.constituents.top_detractors[:5]),
                    "evidence": list(result.analysis.constituents.evidence),
                },
                "options": {
                    "max_pain": (
                        float(result.analysis.options.max_pain_strike)
                        if result.analysis.options.max_pain_strike is not None
                        else None
                    ),
                    "call_walls": [float(x) for x in result.analysis.options.call_walls],
                    "put_walls": [float(x) for x in result.analysis.options.put_walls],
                    "pcr_oi": result.analysis.options.pcr_oi,
                    "oi_structure_score": result.analysis.options.oi_structure_score,
                    # None, not 0.0, when the forward was never solved.
                    "basis_score": result.analysis.options.basis_score,
                    "excess_basis": (
                        float(result.analysis.options.excess_basis)
                        if result.analysis.options.excess_basis is not None
                        else None
                    ),
                    "confidence": result.analysis.options.confidence,
                },
                "volatility": {
                    "regime": str(result.analysis.volatility.regime),
                    "atm_iv": result.analysis.volatility.atm_iv,
                    "iv_percentile": result.analysis.volatility.iv_percentile,
                    "expected_move": (
                        float(result.analysis.volatility.expected_move)
                        if result.analysis.volatility.expected_move is not None
                        else None
                    ),
                    "expected_absolute_move": (
                        float(result.analysis.volatility.expected_absolute_move)
                        if result.analysis.volatility.expected_absolute_move is not None
                        else None
                    ),
                    "confidence": result.analysis.volatility.confidence,
                },
            },
            "strategy": str(result.selected_strategy),
            "is_actionable": result.is_actionable,
            # Authorization requires an account and a portfolio the system
            # cannot see without a broker, so this is always false for now —
            # and says so rather than being omitted.
            "is_authorized": result.is_authorized,
            "authorization_blocked_reason": (
                None
                if result.risk_decision is not None
                else "No broker connected, so the Risk Engine has no account to size against"
            ),
            "candidate": (
                {
                    "strategy": str(candidate.strategy),
                    # Per lot, matching how the Strike Engine prices a
                    # structure. Risk decides how many lots that becomes, and
                    # without a broker it never gets to.
                    "max_loss_per_lot": float(candidate.max_loss),
                    "max_profit_per_lot": (
                        float(candidate.max_profit)
                        if candidate.max_profit is not None
                        else None
                    ),
                    "net_premium": float(candidate.net_premium),
                    "is_credit": candidate.is_credit,
                    "reward_to_risk": candidate.reward_to_risk,
                    "net_reward_to_risk": candidate.net_reward_to_risk,
                    "round_trip_cost": float(candidate.round_trip_cost),
                    "cost_share_of_profit": candidate.cost_share_of_profit,
                    # The two numbers that decide a buy.
                    "breakeven_sigmas": candidate.breakeven_sigmas,
                    "probability_of_profit": candidate.probability_of_profit,
                    "score": candidate.score,
                    "liquidity_score": candidate.liquidity_score,
                    "worst_relative_spread": candidate.worst_relative_spread,
                    "breakeven": [float(level) for level in candidate.breakeven],
                    "rationale": candidate.rationale,
                    "legs": [
                        {
                            "strike": float(leg.contract.strike),
                            "option_type": str(leg.contract.option_type),
                            "side": str(leg.side),
                            "lots": leg.lots,
                            "reference_price": float(leg.reference_price),
                            "delta": (
                                float(leg.delta) if leg.delta is not None else None
                            ),
                        }
                        for leg in candidate.legs
                    ],
                }
                if candidate is not None
                else None
            ),
        }

    @app.get("/api/history/{symbol}")
    async def history(symbol: str, limit: int = 60) -> dict[str, Any]:
        """Recorded analysis cycles, newest first.

        Reads the database rather than a ring buffer in memory, so the panel
        survives a restart — which is the point of persisting cycles at all.
        Returns `available: False` with a reason when capture is off, never
        an empty list that would read as "the engine has decided nothing".
        """
        symbol = symbol.upper()
        if live.capture is None:
            return {
                "available": False,
                "symbol": symbol,
                "reason": "Capture is disabled, so no history is being recorded",
                "cycles": [],
            }
        cycles = await live.recent_cycles(symbol, limit=min(limit, 500))
        return {
            "available": True,
            "symbol": symbol,
            "count": len(cycles),
            "cycles": cycles,
        }

    @app.get("/", response_class=HTMLResponse)
    def console() -> HTMLResponse:
        if not CONSOLE_HTML.exists():
            return HTMLResponse(
                "<h1>Console not found</h1>"
                f"<p>Expected {CONSOLE_HTML}</p>",
                status_code=404,
            )
        return HTMLResponse(CONSOLE_HTML.read_text())

    return app


def _decimal_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


class DecimalJSONResponse(JSONResponse):
    """Kept for routes that hand back raw contract objects."""

    def render(self, content: Any) -> bytes:
        import json

        return json.dumps(content, default=_decimal_default).encode()


app = create_app()
