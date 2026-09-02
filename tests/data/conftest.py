"""Recorded-payload plumbing for live adapter tests.

The transport is replaced; the market data is not. Every payload below is a
response NSE actually sent (see `recorded/README.md`), so these tests assert
that the adapter reads the real exchange correctly rather than that it reads
a convenient fiction correctly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from index_option_brain.data.adapters.nse_public import NsePublicAdapter
from index_option_brain.data.http import HttpResponse, RecordedSession

RECORDED = Path(__file__).parent / "recorded"


def payload(name: str) -> str:
    return (RECORDED / name).read_text()


def parsed(name: str) -> Any:
    return json.loads(payload(name))


NSE_ROUTES = {
    "/api/allIndices": payload("nse_all_indices.json"),
    "/api/option-chain-contract-info": payload("nse_contract_info.json"),
    "/api/option-chain-v3": payload("nse_option_chain.json"),
}


def nse_session(**overrides: HttpResponse | str) -> RecordedSession:
    """A recorded NSE session, with per-test route overrides taking priority.

    RecordedSession matches route keys as substrings in insertion order, so
    overrides are inserted **first** — otherwise a test that overrides
    `allIndices` with a 500 would still be served the healthy payload by the
    broader default route, and would pass for the wrong reason.
    """
    routes: dict[str, HttpResponse | str] = dict(overrides)
    routes.update(NSE_ROUTES)
    # The warm-up hits the page, not the API, so it needs its own route.
    routes["https://www.nseindia.com/option-chain"] = "<html>option chain page</html>"
    return RecordedSession(routes)


@pytest.fixture
def session() -> RecordedSession:
    return nse_session()


@pytest.fixture
def nse(session: RecordedSession) -> NsePublicAdapter:
    return NsePublicAdapter(session)
