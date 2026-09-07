"""The registry, and the two credentials every endpoint has."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from index_option_brain.integrations.webhooks.endpoints import (
    Endpoint,
    EndpointKind,
    load,
)

INGEST = "ingest-secret-long-enough"
READ = "read-token-long-enough-too"


class TestCredentials:
    def test_the_two_credentials_must_differ(self) -> None:
        """The pusher's secret sits in a TradingView indicator input that
        every viewer of a shared chart can read. Sharing it with the read
        token would let anyone who sees the chart read every delivery."""
        with pytest.raises(ValueError, match="must differ"):
            Endpoint(slug="tv", ingest_secret=INGEST, read_token=INGEST)

    @pytest.mark.parametrize("field", ["ingest_secret", "read_token"])
    def test_a_short_credential_is_refused(self, field: str) -> None:
        kwargs = {"slug": "tv", "ingest_secret": INGEST, "read_token": READ, field: "short"}
        with pytest.raises(ValueError, match="at least 16"):
            Endpoint(**kwargs)  # type: ignore[arg-type]


class TestSlug:
    @pytest.mark.parametrize("slug", ["My Hook", "hook/../etc", "hôk", "", "UPPER"])
    def test_an_unsafe_slug_is_refused_not_sanitised(self, slug: str) -> None:
        """Quietly rewriting `My Hook` to `my-hook` means the URL you were
        given and the URL that works are different strings."""
        with pytest.raises(ValueError, match="slug"):
            Endpoint(slug=slug, ingest_secret=INGEST, read_token=READ)

    def test_a_safe_slug_passes(self) -> None:
        assert Endpoint(slug="tv_nifty-1", ingest_secret=INGEST, read_token=READ).slug


class TestLoading:
    def _write(self, tmp_path: Path, document: dict[str, object], mode: int = 0o600) -> Path:
        file = tmp_path / "endpoints.json"
        file.write_text(json.dumps(document))
        file.chmod(mode)
        return file

    def test_a_missing_file_is_an_empty_registry(self, tmp_path: Path) -> None:
        assert load(tmp_path / "nothing.json") == {}

    def test_it_reads_endpoints(self, tmp_path: Path) -> None:
        file = self._write(
            tmp_path,
            {
                "tradingview": {
                    "ingest_secret": INGEST,
                    "read_token": READ,
                    "kind": "tradingview",
                    "retain": 100,
                    "allowed_ips": "52.89.214.238, 34.212.75.30",
                }
            },
        )
        endpoints = load(file)
        assert endpoints["tradingview"].kind is EndpointKind.TRADINGVIEW
        assert endpoints["tradingview"].retain == 100
        assert endpoints["tradingview"].allowed_ips == frozenset(
            {"52.89.214.238", "34.212.75.30"}
        )

    def test_a_world_readable_registry_is_refused(self, tmp_path: Path) -> None:
        """It holds credentials, on a box that also runs an assistant with
        shell access. That is not a warning-level problem."""
        file = self._write(tmp_path, {"tv": {"ingest_secret": INGEST, "read_token": READ}}, mode=0o644)
        with pytest.raises(PermissionError, match="credentials"):
            load(file)

    def test_secrets_can_live_in_the_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """So the registry can be committed while the secrets are not."""
        monkeypatch.setenv("TV_INGEST", INGEST)
        monkeypatch.setenv("TV_READ", READ)
        file = self._write(
            tmp_path, {"tv": {"ingest_secret": "${TV_INGEST}", "read_token": "${TV_READ}"}}
        )
        assert load(file)["tv"].ingest_secret == INGEST

    def test_a_non_object_document_is_refused(self, tmp_path: Path) -> None:
        file = tmp_path / "endpoints.json"
        file.write_text("[]")
        file.chmod(0o600)
        with pytest.raises(ValueError, match="JSON object"):
            load(file)
