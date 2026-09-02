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

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from decimal import Decimal
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from index_option_brain.app.live import FeedUnavailable, LiveEngine, session_label
from index_option_brain.config.settings import get_settings
from index_option_brain.contracts.provider import Capability, ProviderDescriptor
from index_option_brain.data.providers import (
    ALL_PROVIDERS,
    REQUIRED_FOR_ANALYSIS,
    REQUIRED_FOR_TRADING,
    implemented_providers,
    missing_capabilities,
)

CONSOLE_HTML = Path(__file__).resolve().parents[2] / "docs" / "console.html"


def _capability_names(capabilities: frozenset[Capability]) -> list[str]:
    return sorted(str(capability) for capability in capabilities)


def _describe(provider: ProviderDescriptor, health: Any) -> dict[str, Any]:
    return {
        "provider_id": provider.provider_id,
        "display_name": provider.display_name,
        "kind": str(provider.kind),
        "auth": str(provider.auth),
        "implemented": provider.implemented,
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


def create_app(engine: LiveEngine | None = None) -> FastAPI:
    settings = get_settings()
    live = engine or LiveEngine()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await live.aclose()

    app = FastAPI(title="Index Option Brain", version="0.1.0", lifespan=lifespan)
    app.state.live = live

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "llm_enabled": settings.llm_enabled,
            "run_mode": settings.run_mode,
        }

    @app.get("/api/status")
    def status() -> dict[str, object]:
        """What the system is configured to do, and what it cannot do yet."""
        connected = implemented_providers()
        return {
            "run_mode": str(settings.run_mode),
            "llm_enabled": settings.llm_enabled,
            "kill_switch_engaged": settings.kill_switch_enabled,
            "trading_enabled": False,
            "trading_blocked_reason": (
                "No broker adapter is implemented, so no order can be placed. "
                "The system is analysis-only until one is connected."
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
            "breadth": {
                "constituents": len(state.constituent_state.quotes),
                "available": bool(state.constituent_state.quotes),
                "reason": (
                    None
                    if state.constituent_state.quotes
                    else "No connected provider serves index constituents"
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
