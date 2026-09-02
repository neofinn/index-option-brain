"""FastAPI application factory. Deliberately minimal at this stage — a
health/readiness endpoint and settings wiring only. Routes for triggering
analysis, inspecting state, and (later) driving paper/live trading get added
alongside the engines that back them."""

from __future__ import annotations

from fastapi import FastAPI

from index_option_brain.config.settings import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Index Option Brain", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "llm_enabled": settings.llm_enabled,
            "run_mode": settings.run_mode,
        }

    return app


app = create_app()
