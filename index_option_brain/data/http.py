"""A minimal async HTTP seam for live data adapters.

Adapters depend on the `HttpSession` protocol rather than on a concrete
client, for one reason that matters more than testability: the fragile part
of a live adapter is *payload parsing*, and parsing can only be tested
deterministically if the transport can be replaced by recorded responses.
Wiring `httpx` directly into an adapter would make every parsing test a
network test, which is exactly the kind of test that gets deleted the first
time an exchange is closed.

The protocol is deliberately tiny — one GET and a close. Live market data
adapters read; anything that writes belongs behind a broker adapter with its
own authentication contract.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, Self, runtime_checkable

DEFAULT_TIMEOUT_SECONDS = 10.0


class HttpError(RuntimeError):
    """A transport-level failure: connection refused, timeout, TLS error.

    Kept distinct from a non-200 response, which is a fact about the server
    and is reported through `HttpResponse.status_code` instead. An adapter
    needs to tell "the exchange said no" apart from "we could not reach the
    exchange", because only the first tells you anything about the market.
    """


@dataclass(frozen=True)
class HttpResponse:
    status_code: int
    text: str
    headers: Mapping[str, str] = field(default_factory=dict)

    @property
    def is_ok(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self) -> Any:
        """Parse the body as JSON, or raise `ValueError`.

        Callers must handle the failure rather than assume JSON: public
        exchange endpoints answer an unauthenticated or rate-limited request
        with an HTML interstitial and a 200 status, and treating that as data
        is how a scraper silently starts reporting nonsense.
        """
        return json.loads(self.text)


@runtime_checkable
class HttpSession(Protocol):
    """A cookie-persisting HTTP session.

    Cookie persistence across calls is part of the contract, not an
    implementation detail: NSE's public API rejects a request that does not
    carry the cookies its HTML pages set, so an adapter's warm-up request is
    only meaningful if the session remembers it.
    """

    async def get(
        self,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse: ...

    async def post(
        self,
        url: str,
        *,
        json: Any | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse: ...

    async def delete(
        self, url: str, *, headers: Mapping[str, str] | None = None
    ) -> HttpResponse: ...

    async def aclose(self) -> None: ...


def _load_httpx() -> Any:
    """Import httpx, falling back to the `httpx2` distribution name.

    `httpx` is the declared dependency. Some environments ship the same
    library packaged as `httpx2`, and accepting either keeps a live adapter
    runnable there rather than failing at import over a package name.
    """
    for module_name in ("httpx", "httpx2"):
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
    raise HttpError(
        "No httpx installation found. Install `httpx` to use live HTTP "
        "adapters, or inject an HttpSession implementation instead."
    )


class HttpxSession:
    """`HttpSession` backed by httpx, with cookies kept across requests.

    `trust_env` is left on so the process honours proxy and CA-bundle
    environment variables. Certificate verification is never disabled: a
    market data feed whose identity is unverified is not a market data feed.
    """

    def __init__(
        self,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        httpx = _load_httpx()
        self._httpx = httpx
        self._client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            trust_env=True,
            headers=dict(headers or {}),
        )

    async def get(
        self,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        try:
            response = await self._client.get(
                url,
                params=dict(params) if params else None,
                headers=dict(headers) if headers else None,
            )
        # Broad by necessity: httpx is imported dynamically, so its exception
        # tree is not available to name here. Every transport failure is
        # normalized to HttpError so adapters have one thing to catch.
        except Exception as exc:
            raise HttpError(f"GET {url} failed: {type(exc).__name__}: {exc}") from exc
        return HttpResponse(
            status_code=response.status_code,
            text=response.text,
            headers=dict(response.headers),
        )

    async def post(
        self,
        url: str,
        *,
        json: Any | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        try:
            response = await self._client.post(
                url, json=json, headers=dict(headers) if headers else None
            )
        except Exception as exc:
            raise HttpError(f"POST {url} failed: {type(exc).__name__}: {exc}") from exc
        return HttpResponse(
            status_code=response.status_code,
            text=response.text,
            headers=dict(response.headers),
        )

    async def delete(
        self, url: str, *, headers: Mapping[str, str] | None = None
    ) -> HttpResponse:
        try:
            response = await self._client.delete(
                url, headers=dict(headers) if headers else None
            )
        except Exception as exc:
            raise HttpError(f"DELETE {url} failed: {type(exc).__name__}: {exc}") from exc
        return HttpResponse(
            status_code=response.status_code,
            text=response.text,
            headers=dict(response.headers),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()


class RecordedSession:
    """An `HttpSession` that replays canned responses, keyed by URL substring.

    Used by adapter parsing tests with payloads captured from the real
    endpoint. It is a test double for the *transport*, not for the market:
    the bodies it returns are recorded exchange responses, so a parsing test
    built on it asserts against data the exchange actually sent.
    """

    def __init__(self, routes: Mapping[str, HttpResponse | str]) -> None:
        self._routes = {
            key: value if isinstance(value, HttpResponse) else HttpResponse(200, value)
            for key, value in routes.items()
        }
        self.requests: list[str] = []
        self.deleted: list[str] = []
        self.posted: list[tuple[str, Any]] = []
        """Every POST with its body, so a test can assert what was asked for
        as well as what came back — which is where a wrong segment code or a
        mis-ordered date range shows up."""

    async def get(
        self,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        full = url
        if params:
            full = f"{url}?" + "&".join(f"{k}={v}" for k, v in params.items())
        self.requests.append(full)
        for key, response in self._routes.items():
            if key in full:
                return response
        raise HttpError(f"RecordedSession has no route matching {full}")

    async def post(
        self,
        url: str,
        *,
        json: Any | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.posted.append((url, json))
        self.requests.append(url)
        for key, response in self._routes.items():
            if key in url:
                return response
        raise HttpError(f"RecordedSession has no route matching {url}")

    async def delete(
        self, url: str, *, headers: Mapping[str, str] | None = None
    ) -> HttpResponse:
        self.deleted.append(url)
        self.requests.append(url)
        for key, response in self._routes.items():
            if key in url:
                return response
        raise HttpError(f"RecordedSession has no route matching {url}")

    async def aclose(self) -> None:
        return None
