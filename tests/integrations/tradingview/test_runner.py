"""What the receiver process refuses to do at startup."""

from __future__ import annotations

import pytest

from index_option_brain.config.settings import Settings
from index_option_brain.integrations.tradingview.__main__ import (
    allowed_ips,
    build_config,
    main,
)
from index_option_brain.integrations.tradingview.auth import TRADINGVIEW_EGRESS_IPS


class TestAllowlistParsing:
    def test_empty_means_tradingviews_published_set(self) -> None:
        assert allowed_ips("") == TRADINGVIEW_EGRESS_IPS

    def test_any_is_the_explicit_opt_out(self) -> None:
        """Needed behind a tunnel, where the peer address is the tunnel.
        Spelled out so it is a decision rather than an oversight."""
        assert allowed_ips("any") == frozenset()
        assert allowed_ips("ANY") == frozenset()

    def test_a_list_is_parsed_and_trimmed(self) -> None:
        assert allowed_ips(" 1.2.3.4 , 5.6.7.8 ") == frozenset({"1.2.3.4", "5.6.7.8"})


class TestStartup:
    def test_it_refuses_to_start_without_a_secret(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An open POST endpoint on the public internet that anyone who
        finds the URL can push market claims into is worse than no
        receiver."""
        monkeypatch.setattr(
            "index_option_brain.integrations.tradingview.__main__.get_settings",
            lambda: Settings(TRADINGVIEW_WEBHOOK_SECRET=""),
        )
        assert main() == 2

    def test_it_refuses_to_start_with_a_guessable_secret(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "index_option_brain.integrations.tradingview.__main__.get_settings",
            lambda: Settings(TRADINGVIEW_WEBHOOK_SECRET="changeme"),
        )
        assert main() == 2

    def test_a_real_secret_builds_a_guard_config(self) -> None:
        config = build_config(
            Settings(TRADINGVIEW_WEBHOOK_SECRET="a-sufficiently-long-secret")
        )
        assert config.allowed_ips == TRADINGVIEW_EGRESS_IPS
        assert config.trust_forwarded_for is False
