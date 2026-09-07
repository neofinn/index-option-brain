"""The conversion itself: a POST goes in, a GET comes out."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from index_option_brain.database.engine import Database
from index_option_brain.integrations.webhooks.endpoints import Endpoint, EndpointKind
from index_option_brain.integrations.webhooks.gateway import (
    MAX_BODY_BYTES,
    RateLimiter,
    create_gateway_app,
    parse_body,
    strip_credentials,
)
from index_option_brain.integrations.webhooks.store import DeliveryStore

INGEST = "ingest-secret-long-enough"
READ = "read-token-long-enough-too"
OTHER_READ = "another-read-token-here-ok"
NOW = datetime(2026, 9, 7, 4, 15, tzinfo=UTC)


@pytest.fixture
async def database() -> AsyncIterator[Database]:
    db = Database.in_memory()
    await db.create_schema()
    yield db
    await db.aclose()


@pytest.fixture
def registry() -> dict[str, Endpoint]:
    return {
        "tv": Endpoint(slug="tv", ingest_secret=INGEST, read_token=READ, retain=3),
        "other": Endpoint(
            slug="other", ingest_secret=INGEST + "-2", read_token=OTHER_READ
        ),
    }


@pytest.fixture
def client(database: Database, registry: dict[str, Endpoint]) -> TestClient:
    app = create_gateway_app(
        registry, DeliveryStore(database), clock=lambda: NOW
    )
    return TestClient(app)


def post(client: TestClient, slug: str = "tv", **body: object) -> object:
    payload: dict[str, object] = {"secret": INGEST, "hello": "world"}
    payload.update(body)
    return client.post(f"/hook/{slug}", content=json.dumps(payload))


class TestPushToPull:
    def test_a_delivery_becomes_readable_over_the_api(self, client: TestClient) -> None:
        assert post(client).status_code == 200
        body = client.get("/v1/tv", headers={"Authorization": f"Bearer {READ}"}).json()
        assert body["count"] == 1
        assert body["deliveries"][0]["payload"]["hello"] == "world"

    def test_the_cursor_never_skips_or_repeats(self, client: TestClient) -> None:
        """An integer cursor is exact, unlike a timestamp one — two
        deliveries in the same second are still two distinct positions."""
        head = {"Authorization": f"Bearer {READ}"}
        for n in range(3):
            post(client, n=n)
        first = client.get("/v1/tv?limit=2", headers=head).json()
        assert [d["payload"]["n"] for d in first["deliveries"]] == [0, 1]
        second = client.get(f"/v1/tv?since={first['next_cursor']}", headers=head).json()
        assert [d["payload"]["n"] for d in second["deliveries"]] == [2]
        third = client.get(f"/v1/tv?since={second['next_cursor']}", headers=head).json()
        assert third["deliveries"] == []

    def test_an_empty_page_echoes_the_cursor_back(self, client: TestClient) -> None:
        """So a poller stores one value unconditionally instead of
        branching on an empty page and accidentally rewinding."""
        head = {"Authorization": f"Bearer {READ}"}
        post(client)
        first = client.get("/v1/tv", headers=head).json()
        again = client.get(f"/v1/tv?since={first['next_cursor']}", headers=head).json()
        assert again["next_cursor"] == first["next_cursor"]

    def test_latest_returns_the_newest(self, client: TestClient) -> None:
        post(client, n=1)
        post(client, n=2)
        body = client.get(
            "/v1/tv/latest", headers={"Authorization": f"Bearer {READ}"}
        ).json()
        assert body["delivery"]["payload"]["n"] == 2

    def test_nothing_yet_is_an_empty_state_not_a_404(self, client: TestClient) -> None:
        """A 404 would make "nothing has arrived" indistinguishable from a
        wrong URL, which is the first thing anyone gets wrong."""
        response = client.get("/v1/tv/latest", headers={"Authorization": f"Bearer {READ}"})
        assert response.status_code == 200
        assert response.json()["delivery"] is None

    def test_payload_strips_the_envelope_for_a_shell(self, client: TestClient) -> None:
        post(client, price="24025.65")
        body = client.get(
            "/v1/tv/payload", headers={"Authorization": f"Bearer {READ}"}
        ).json()
        assert body["price"] == "24025.65"
        assert "seq" not in body


class TestIngestAuth:
    def test_a_wrong_secret_is_refused(self, client: TestClient) -> None:
        assert post(client, secret="nope").status_code == 401

    def test_a_header_secret_works_for_senders_that_can_set_one(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/hook/tv",
            content=json.dumps({"hello": "world"}),
            headers={"X-Webhook-Secret": INGEST},
        )
        assert response.status_code == 200

    def test_an_unknown_slug_is_a_404(self, client: TestClient) -> None:
        """A slug is not a secret — it is in the URL you hand the sender —
        and a gateway that answers every mistake identically is one nobody
        can debug."""
        assert client.post("/hook/nope", content="{}").status_code == 404

    def test_an_oversized_body_is_refused(self, client: TestClient) -> None:
        assert client.post("/hook/tv", content="x" * (MAX_BODY_BYTES + 1)).status_code == 413

    def test_an_ip_allowlist_is_enforced(self, database: Database) -> None:
        app = create_gateway_app(
            {
                "tv": Endpoint(
                    slug="tv",
                    ingest_secret=INGEST,
                    read_token=READ,
                    allowed_ips=frozenset({"203.0.113.7"}),
                )
            },
            DeliveryStore(database),
            clock=lambda: NOW,
        )
        client = TestClient(app)
        assert post(client).status_code == 401


class TestReadAuth:
    def test_a_read_token_grants_only_its_own_endpoint(self, client: TestClient) -> None:
        """One leaked credential should cost one endpoint, not the whole
        gateway. There is no token that reads everything."""
        assert client.get("/v1/other", headers={"Authorization": f"Bearer {READ}"}).status_code == 401
        assert client.get("/v1/tv", headers={"Authorization": f"Bearer {OTHER_READ}"}).status_code == 401

    def test_no_token_is_refused(self, client: TestClient) -> None:
        assert client.get("/v1/tv").status_code == 401

    def test_the_ingest_secret_does_not_open_the_read_api(self, client: TestClient) -> None:
        assert client.get("/v1/tv", headers={"Authorization": f"Bearer {INGEST}"}).status_code == 401

    def test_a_query_token_works_for_a_browser(self, client: TestClient) -> None:
        post(client)
        assert client.get(f"/v1/tv?token={READ}").status_code == 200

    def test_the_endpoint_list_shows_only_what_the_token_reads(
        self, client: TestClient
    ) -> None:
        body = client.get("/v1/endpoints", headers={"Authorization": f"Bearer {READ}"}).json()
        assert [e["slug"] for e in body["endpoints"]] == ["tv"]

    def test_the_endpoint_list_reports_how_stale_each_one_is(
        self, database: Database, registry: dict[str, Endpoint]
    ) -> None:
        """A sender that has stopped calling looks exactly like a quiet one
        until you can see how long it has been quiet."""
        later = NOW + timedelta(minutes=20)
        clock = [NOW]
        app = create_gateway_app(
            registry, DeliveryStore(database), clock=lambda: clock[0]
        )
        client = TestClient(app)
        post(client)
        clock[0] = later
        body = client.get("/v1/endpoints", headers={"Authorization": f"Bearer {READ}"}).json()
        assert body["endpoints"][0]["seconds_since_last"] == 1200


class TestCredentialsNeverStored:
    def test_a_body_secret_is_stripped_before_storage(self, client: TestClient) -> None:
        """TradingView has nowhere but the body to put its credential.
        Persisting it verbatim would serve it back over the read API,
        turning the weaker credential into the stronger one."""
        post(client)
        body = client.get("/v1/tv", headers={"Authorization": f"Bearer {READ}"}).json()
        assert INGEST not in json.dumps(body)
        assert "secret" not in body["deliveries"][0]["payload"]

    def test_nested_credentials_are_stripped_too(self) -> None:
        cleaned = strip_credentials(
            {"auth": {"token": "xyz"}, "items": [{"api_key": "k", "id": 3}], "id": 1}
        )
        assert cleaned == {"items": [{"id": 3}], "id": 1}

    def test_a_field_that_merely_contains_the_word_survives(self) -> None:
        """`secret_santa_id` is content. Dropping it would silently lose
        data the consumer needs."""
        assert strip_credentials({"secret_santa_id": 7}) == {"secret_santa_id": 7}


class TestBodyShapes:
    def test_json_from_a_text_plain_sender_is_still_parsed(self) -> None:
        """TradingView sends JSON as text/plain. Routing on the declared
        content type would put every alert in the text branch."""
        assert parse_body(b'{"a":1}', "text/plain") == {"a": 1}

    def test_an_array_is_wrapped_so_a_row_is_always_an_object(self) -> None:
        assert parse_body(b"[1,2]", None) == {"value": [1, 2]}

    def test_form_encoding_is_parsed(self) -> None:
        assert parse_body(
            b"a=1&b=two", "application/x-www-form-urlencoded"
        ) == {"a": "1", "b": "two"}

    def test_an_unparseable_body_is_kept_as_text(self) -> None:
        """A body this service cannot parse is still a delivery that
        happened. Losing it would hide the sender's misconfiguration."""
        assert parse_body(b"BUY NIFTY NOW", None) == {"text": "BUY NIFTY NOW"}

    def test_invalid_utf8_does_not_raise(self) -> None:
        assert "text" in parse_body(b"\xff\xfe binary", None)


class TestRetentionAndRate:
    def test_only_the_newest_are_kept(self, client: TestClient) -> None:
        """A sender misconfigured to fire every second fills a disk in a
        day, and the failure lands on the engine sharing that disk."""
        for n in range(6):
            post(client, n=n)
        body = client.get("/v1/tv", headers={"Authorization": f"Bearer {READ}"}).json()
        assert [d["payload"]["n"] for d in body["deliveries"]] == [3, 4, 5]

    def test_a_flood_is_rate_limited(self, database: Database, registry: dict[str, Endpoint]) -> None:
        app = create_gateway_app(
            registry,
            DeliveryStore(database),
            rate_limiter=RateLimiter(limit=2, clock=lambda: 0.0),
            clock=lambda: NOW,
        )
        client = TestClient(app)
        assert post(client).status_code == 200
        assert post(client).status_code == 200
        assert post(client).status_code == 429

    def test_the_limit_is_per_endpoint(self, database: Database, registry: dict[str, Endpoint]) -> None:
        """One noisy sender must not lock out a quiet one."""
        app = create_gateway_app(
            registry,
            DeliveryStore(database),
            rate_limiter=RateLimiter(limit=1, clock=lambda: 0.0),
            clock=lambda: NOW,
        )
        client = TestClient(app)
        assert post(client, "tv").status_code == 200
        assert post(client, "tv").status_code == 429
        response = client.post(
            "/hook/other", content=json.dumps({"secret": INGEST + "-2"})
        )
        assert response.status_code == 200


class TestSurface:
    def test_health_reveals_no_slugs_or_counts(self, client: TestClient) -> None:
        """It answers the public internet, and what is configured here is
        information about what the operator is watching."""
        body = client.get("/health").json()
        assert body == {"status": "ok", "endpoints": 2}

    def test_the_only_write_route_is_the_hook(self, client: TestClient) -> None:
        writes = {
            (route.path, method)
            for route in client.app.routes  # type: ignore[attr-defined]
            for method in getattr(route, "methods", set())
            if method not in {"GET", "HEAD", "OPTIONS"}
        }
        assert writes == {("/hook/{slug}", "POST")}

    def test_the_page_is_served_as_a_whole_document(self, client: TestClient) -> None:
        """The file on disk is a fragment so it can also be published as an
        artifact; the shell is added here."""
        body = client.get("/").text
        assert body.startswith("<!doctype html>")
        assert "Webhook to API" in body

    def test_no_cors_header_is_offered(self, client: TestClient) -> None:
        """With one, any page anywhere could be made to read this gateway
        using a token its viewer pasted somewhere else."""
        response = client.get("/health")
        assert "access-control-allow-origin" not in {
            k.lower() for k in response.headers
        }


class TestHandler:
    def test_a_raw_endpoint_never_runs_the_handler(self, database: Database) -> None:
        """A raw endpoint validates nothing on purpose: it carries payloads
        whose shape this system does not know."""
        calls: list[str] = []
        app = create_gateway_app(
            {"tv": Endpoint(slug="tv", ingest_secret=INGEST, read_token=READ)},
            DeliveryStore(database),
            clock=lambda: NOW,
            on_delivery=lambda ep, payload: calls.append(ep.slug),
        )
        assert post(TestClient(app)).status_code == 200
        assert calls == []

    def test_a_handler_note_reaches_the_sender(self, database: Database) -> None:
        """A rejection only a server log knows about is one the operator
        finds out about days later."""
        app = create_gateway_app(
            {
                "tv": Endpoint(
                    slug="tv",
                    ingest_secret=INGEST,
                    read_token=READ,
                    kind=EndpointKind.TRADINGVIEW,
                )
            },
            DeliveryStore(database),
            clock=lambda: NOW,
            on_delivery=lambda ep, payload: "NOT_CHART_OBSERVABLE",
        )
        assert post(TestClient(app)).json()["handler"] == "NOT_CHART_OBSERVABLE"

    def test_a_failing_handler_does_not_lose_the_delivery(
        self, database: Database
    ) -> None:
        """It is already durable and readable over the API. A 5xx would
        tell the sender to give up on a payload this service has kept."""
        def boom(ep: Endpoint, payload: dict[str, object]) -> str:
            raise RuntimeError("handler bug")

        app = create_gateway_app(
            {
                "tv": Endpoint(
                    slug="tv",
                    ingest_secret=INGEST,
                    read_token=READ,
                    kind=EndpointKind.TRADINGVIEW,
                )
            },
            DeliveryStore(database),
            clock=lambda: NOW,
            on_delivery=boom,
        )
        client = TestClient(app)
        response = post(client)
        assert response.status_code == 200
        assert response.json()["handler"] == "failed"
        body = client.get("/v1/tv", headers={"Authorization": f"Bearer {READ}"}).json()
        assert body["count"] == 1


class TestBehindAProxy:
    """TradingView calls ports 80 and 443 only, so a proxy is always in
    front — which is what makes the peer address useless for an allowlist.
    """

    def _client(
        self, database: Database, *, trust: bool, allowed: str
    ) -> TestClient:
        app = create_gateway_app(
            {
                "tv": Endpoint(
                    slug="tv",
                    ingest_secret=INGEST,
                    read_token=READ,
                    allowed_ips=frozenset({allowed}),
                )
            },
            DeliveryStore(database),
            clock=lambda: NOW,
            trust_forwarded_for=trust,
        )
        return TestClient(app)

    def test_an_allowlist_behind_a_proxy_rejects_everything_untrusted(
        self, database: Database
    ) -> None:
        """The footgun. The peer is the proxy, so an allowlist of the
        sender's egress addresses matches nothing — and the 401 looks
        exactly like a wrong secret."""
        client = self._client(database, trust=False, allowed="52.89.214.238")
        response = client.post(
            "/hook/tv",
            content=json.dumps({"secret": INGEST}),
            headers={"X-Forwarded-For": "52.89.214.238"},
        )
        assert response.status_code == 401

    def test_a_declared_proxy_lets_the_allowlist_work(
        self, database: Database
    ) -> None:
        client = self._client(database, trust=True, allowed="52.89.214.238")
        response = client.post(
            "/hook/tv",
            content=json.dumps({"secret": INGEST}),
            headers={"X-Forwarded-For": "52.89.214.238, 10.0.0.1"},
        )
        assert response.status_code == 200

    def test_the_leftmost_forwarded_entry_is_the_client(
        self, database: Database
    ) -> None:
        """A proxy appends, so the leftmost value is the original client as
        the first trusted hop saw it."""
        client = self._client(database, trust=True, allowed="10.0.0.1")
        response = client.post(
            "/hook/tv",
            content=json.dumps({"secret": INGEST}),
            headers={"X-Forwarded-For": "52.89.214.238, 10.0.0.1"},
        )
        assert response.status_code == 401

    def test_an_untrusted_header_cannot_forge_an_address(
        self, database: Database
    ) -> None:
        client = self._client(database, trust=False, allowed="testclient")
        response = client.post(
            "/hook/tv",
            content=json.dumps({"secret": INGEST}),
            headers={"X-Forwarded-For": "1.2.3.4"},
        )
        # The peer address still decides, so the allowlist still holds.
        assert response.status_code == 200


class TestPastingTheUrlIntoABrowser:
    def test_a_get_on_the_hook_path_explains_itself(self, client: TestClient) -> None:
        """FastAPI's bare 405 is the least helpful answer available at the
        moment someone is checking whether they copied the URL right."""
        response = client.get("/hook/tv")
        assert response.status_code == 405
        body = response.json()
        assert body["accepts"] == "POST"
        assert "443" in body["detail"]

    def test_an_unknown_slug_still_404s(self, client: TestClient) -> None:
        assert client.get("/hook/nope").status_code == 404

    def test_it_adds_no_write_route(self, client: TestClient) -> None:
        writes = {
            (route.path, method)
            for route in client.app.routes  # type: ignore[attr-defined]
            for method in getattr(route, "methods", set())
            if method not in {"GET", "HEAD", "OPTIONS"}
        }
        assert writes == {("/hook/{slug}", "POST")}
