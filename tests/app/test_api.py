"""The API the console reads.

Every test runs against **recorded NSE payloads** through the same adapter the
live system uses (see `tests/data/recorded/README.md`), so the endpoints are
exercised on data the exchange actually sent, with no network.

The property under test throughout is that a gap is reported as a gap. An API
that quietly substitutes a zero for a missing measurement produces a console
that looks healthy while showing nothing real, which is the specific failure
this system is built to avoid.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from index_option_brain.app.live import LiveEngine
from index_option_brain.app.main import create_app
from index_option_brain.contracts.enums import BarInterval
from index_option_brain.data.adapters.nse_public import NsePublicAdapter
from index_option_brain.data.bar_aggregator import AggregatingIndexAdapter
from index_option_brain.data.http import HttpResponse
from index_option_brain.state.market_state_builder import (
    InMemoryIvHistoryStore,
    MarketStateBuilder,
)
from tests.data.conftest import nse_session


def engine_on(session) -> LiveEngine:
    """A LiveEngine wired to a recorded transport rather than the network."""
    nse = NsePublicAdapter(session)
    index = AggregatingIndexAdapter(nse, intervals=(BarInterval.MINUTE_5, BarInterval.DAY))
    engine = LiveEngine(cache_seconds=0.0)
    engine._nse = nse
    engine._index = index
    engine._builder = MarketStateBuilder(
        index, None, nse, nse, InMemoryIvHistoryStore()
    )
    return engine


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(engine_on(nse_session()), run_poller=False))


@pytest.fixture
def blocked_client() -> TestClient:
    """A feed that answers every API call with the anti-bot page."""
    session = nse_session(
        allIndices=HttpResponse(200, "<html>Access Denied</html>"),
    )
    return TestClient(create_app(engine_on(session), run_poller=False))


class TestStatus:
    def test_it_reports_the_run_mode_and_llm_setting(self, client: TestClient):
        body = client.get("/api/status").json()
        assert body["llm_enabled"] is False
        assert body["run_mode"]

    def test_trading_is_disabled_with_a_stated_reason(self, client: TestClient):
        """No broker adapter exists, and the console must say so rather than
        showing an execution panel that cannot work."""
        body = client.get("/api/status").json()
        assert body["trading_enabled"] is False
        assert "No broker adapter" in body["trading_blocked_reason"]

    def test_coverage_names_what_is_missing(self, client: TestClient):
        body = client.get("/api/status").json()
        assert body["coverage"]["analysis"]["missing"] == ["INDEX_BARS"]
        assert set(body["coverage"]["trading"]["missing"]) == {
            "ACCOUNT_SNAPSHOT",
            "ORDER_PLACEMENT",
            "POSITION_BOOK",
        }


class TestProviders:
    def test_it_lists_every_provider(self, client: TestClient):
        body = client.get("/api/providers").json()
        assert body["total_count"] == len(body["providers"])
        assert body["implemented_count"] == 1

    def test_unprobed_health_is_not_configured_rather_than_zeroed(
        self, client: TestClient
    ):
        """A console showing 0 ms for a provider never called would be
        reporting a measurement it does not have."""
        body = client.get("/api/providers").json()
        nse = next(p for p in body["providers"] if p["provider_id"] == "nse_public")
        assert nse["health"]["state"] == "NOT_CONFIGURED"
        assert nse["health"]["latency_ms"] is None
        assert nse["health"]["verified_capabilities"] == []

    def test_probing_verifies_capabilities_by_calling_them(self, client: TestClient):
        body = client.get("/api/providers?probe=true").json()
        nse = next(p for p in body["providers"] if p["provider_id"] == "nse_public")
        assert nse["health"]["state"] == "CONNECTED"
        assert nse["health"]["latency_ms"] is not None
        assert set(nse["health"]["verified_capabilities"]) == {
            "EXPIRY_LIST",
            "INDEX_QUOTE",
            "INDIA_VIX",
            "OPTION_CHAIN",
        }

    def test_a_blocked_feed_probes_as_failed_with_the_reason(
        self, blocked_client: TestClient
    ):
        body = blocked_client.get("/api/providers?probe=true").json()
        nse = next(p for p in body["providers"] if p["provider_id"] == "nse_public")
        assert nse["health"]["state"] in {"FAILED", "DEGRADED"}
        assert nse["health"]["last_error"]
        assert not nse["health"]["usable"] or nse["health"]["state"] == "DEGRADED"

    def test_the_listing_still_renders_when_the_probe_fails(
        self, blocked_client: TestClient
    ):
        """The one screen that should always be readable must not break
        because the thing it is reporting on is broken."""
        response = blocked_client.get("/api/providers?probe=true")
        assert response.status_code == 200
        assert len(response.json()["providers"]) > 1

    def test_roadmap_providers_are_marked_unimplemented(self, client: TestClient):
        body = client.get("/api/providers").json()
        kite = next(p for p in body["providers"] if p["provider_id"] == "zerodha_kite")
        assert kite["implemented"] is False
        assert any("not been verified" in note for note in kite["notes"])

    def test_credential_fields_are_carried_for_the_console_to_render(
        self, client: TestClient
    ):
        body = client.get("/api/providers").json()
        angel = next(p for p in body["providers"] if p["provider_id"] == "angel_one")
        names = {field["name"] for field in angel["credential_fields"]}
        assert "totp_secret" in names
        totp = next(f for f in angel["credential_fields"] if f["name"] == "totp_secret")
        assert totp["secret"] is True

    def test_a_data_provider_is_not_offered_as_an_execution_route(
        self, client: TestClient
    ):
        body = client.get("/api/providers").json()
        nse = next(p for p in body["providers"] if p["provider_id"] == "nse_public")
        assert nse["can_trade"] is False
        assert nse["capabilities"]["trading"] == []


class TestMarket:
    def test_it_serves_the_live_index_snapshot(self, client: TestClient):
        body = client.get("/api/market/NIFTY").json()
        assert body["available"] is True
        assert body["index"]["ltp"] == pytest.approx(23914.45)
        assert body["index"]["previous_close"] == pytest.approx(24055.8)
        assert body["index"]["change_pct"] == pytest.approx(-0.5876, abs=1e-3)

    def test_a_value_the_provider_does_not_publish_is_null_not_zero(
        self, client: TestClient
    ):
        """NSE publishes no index VWAP. A zero here would flow into the VWAP
        relationship the Index brain reads and put price permanently above
        it."""
        body = client.get("/api/market/NIFTY").json()
        assert body["index"]["vwap"] is None

    def test_realized_volatility_is_null_without_daily_bars(self, client: TestClient):
        body = client.get("/api/market/NIFTY").json()
        assert body["volatility"]["realized_volatility"] is None

    def test_it_serves_india_vix(self, client: TestClient):
        body = client.get("/api/market/NIFTY").json()
        assert body["volatility"]["india_vix"] == pytest.approx(11.34)
        assert body["volatility"]["india_vix_previous_close"] == pytest.approx(11.49)

    def test_the_expiry_weekday_is_reported(self, client: TestClient):
        """Weekly index expiry moved to Tuesday, and an operator checking the
        console against their broker needs to see which day it is."""
        body = client.get("/api/market/NIFTY").json()
        assert body["options"]["expiry"] == "2026-09-08"
        assert body["options"]["expiry_weekday"] == "Tuesday"

    def test_unmarkable_strikes_are_counted_separately(self, client: TestClient):
        """Strikes whose book is too wide to mark carry no IV and no greeks.
        Reporting them as a count keeps the gap visible instead of making the
        chain look thinner than it is."""
        body = client.get("/api/market/NIFTY").json()
        options = body["options"]
        assert options["legs"] == options["legs_with_greeks"] + options["legs_unmarkable"]
        assert options["legs_unmarkable"] > 0

    def test_bar_coverage_is_reported(self, client: TestClient):
        """A short series and a gappy one look identical in a chart and mean
        different things to an indicator."""
        body = client.get("/api/market/NIFTY").json()
        assert "5m" in body["bars"]
        assert body["bars"]["1d"]["bars"] == 0
        assert body["bars"]["1d"]["has_gaps"] is False

    def test_breadth_reports_its_absence_with_a_reason(self, client: TestClient):
        body = client.get("/api/market/NIFTY").json()
        assert body["breadth"]["available"] is False
        assert body["breadth"]["constituents"] == 0
        assert "No connected provider" in body["breadth"]["reason"]

    def test_the_timestamp_is_the_feeds_own(self, client: TestClient):
        """Not the server's clock: a delayed snapshot must not look fresh."""
        body = client.get("/api/market/NIFTY").json()
        assert body["as_of"].startswith("2026-09-02T10:00")

    def test_the_symbol_is_case_insensitive(self, client: TestClient):
        assert client.get("/api/market/nifty").json()["symbol"] == "NIFTY"

    def test_an_unavailable_feed_is_a_normal_response_not_an_error(
        self, blocked_client: TestClient
    ):
        """A blocked feed is information the operator needs displayed in
        place. Burying it in an error page breaks the one screen that should
        always be readable."""
        response = blocked_client.get("/api/market/NIFTY")
        assert response.status_code == 200
        body = response.json()
        assert body["available"] is False
        assert body["reason"]

    def test_an_unconfigured_symbol_is_reported_not_raised(self, client: TestClient):
        response = client.get("/api/market/FINNIFTY")
        assert response.status_code == 200
        assert response.json()["available"] is False


class TestAnalysis:
    def test_it_runs_the_brain_on_live_state(self, client: TestClient):
        body = client.get("/api/analysis/NIFTY").json()
        assert body["available"] is True
        assert body["strategy"]
        assert "direction" in body["signal"]

    def test_with_no_bars_the_regime_is_uncertain_with_a_reason(
        self, client: TestClient
    ):
        """The console must not show a confident classification of a market
        nothing has measured."""
        body = client.get("/api/analysis/NIFTY").json()
        assert body["regime"]["type"] == "UNCERTAIN"
        assert body["regime"]["confidence"] == 0.0
        assert any("coverage" in line for line in body["regime"]["evidence"])

    def test_nothing_is_authorized_without_a_broker(self, client: TestClient):
        """Authorizing a size against an account the system cannot see would
        be the worst possible invention."""
        body = client.get("/api/analysis/NIFTY").json()
        assert body["is_authorized"] is False
        assert "No broker connected" in body["authorization_blocked_reason"]

    def test_the_candidate_scores_are_reported_for_inspection(
        self, client: TestClient
    ):
        """Refusing to classify is not refusing to explain."""
        body = client.get("/api/analysis/NIFTY").json()
        assert body["regime"]["scores"]

    def test_an_unavailable_feed_is_reported_in_place(self, blocked_client: TestClient):
        response = blocked_client.get("/api/analysis/NIFTY")
        assert response.status_code == 200
        assert response.json()["available"] is False


class TestConsolePage:
    def test_the_console_is_served(self, client: TestClient):
        response = client.get("/")
        assert response.status_code == 200
        assert "Index Brain Console" in response.text

    def test_the_page_carries_no_baked_in_market_figures(self, client: TestClient):
        """The console's whole premise. If a number from a real session ever
        gets pasted into the markup as a placeholder, this catches it."""
        html = client.get("/").text
        for figure in ("23,914", "23914.45", "11.34", "24,055"):
            assert figure not in html, f"{figure} is hard-coded in the console"

    def test_the_page_states_that_its_figures_are_live(self, client: TestClient):
        html = client.get("/").text
        assert "Nothing is sampled or simulated" in html
