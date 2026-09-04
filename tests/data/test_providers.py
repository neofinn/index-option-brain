"""Provider descriptors and the registry.

What is being protected here is a distinction, not a data structure: a
capability read off documentation is a claim, and a capability probed against
a live endpoint is a fact. A trading system that treats them alike will offer
an operator a control that cannot work.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from index_option_brain.contracts.provider import (
    DATA_CAPABILITIES,
    TRADING_CAPABILITIES,
    AuthMethod,
    Capability,
    ProviderConnectionState,
    ProviderHealth,
    ProviderKind,
)
from index_option_brain.data.adapters.nse_public import NSE_PUBLIC_DESCRIPTOR
from index_option_brain.data.providers import (
    ALL_PROVIDERS,
    PROVIDERS_BY_ID,
    REQUIRED_FOR_ANALYSIS,
    REQUIRED_FOR_TRADING,
    ZERODHA_KITE,
    get_provider,
    implemented_providers,
    missing_capabilities,
    providers_serving,
    verified_providers,
)


class TestTheRegistryIsHonest:
    def test_two_providers_have_adapters(self):
        assert {p.provider_id for p in implemented_providers()} == {
            "nse_public",
            "dhan",
        }

    def test_only_nse_public_is_verified_against_real_responses(self):
        """"An adapter exists" and "its mapping was proven" are different
        facts. Dhan's routes and error envelopes were verified; its response
        bodies were not, because those need a token."""
        assert [p.provider_id for p in verified_providers()] == ["nse_public"]

    def test_dhan_says_its_mapping_is_unverified(self):
        kite = get_provider("dhan")
        assert kite.implemented
        assert not kite.verified
        joined = " ".join(kite.notes)
        assert "Response bodies were NOT" in joined
        assert "dhan_probe" in joined

    def test_dhan_records_which_routes_the_sandbox_lacks(self):
        """Probed on both hosts: the sandbox serves orders, funds and charts
        but 404s on market feed and option chain. Learning that from a bare
        404 would be a confusing afternoon."""
        joined = " ".join(get_provider("dhan").notes)
        assert "sandbox" in joined.lower()
        assert "optionchain" in joined

    def test_every_roadmap_entry_says_it_is_unverified(self):
        """Otherwise the registry reads as a list of working integrations."""
        for provider in ALL_PROVIDERS:
            if provider.implemented:
                continue
            joined = " ".join(provider.notes)
            assert "have not been verified" in joined, provider.provider_id
            assert "No adapter exists yet" in joined, provider.provider_id

    def test_no_roadmap_entry_claims_to_be_implemented(self):
        for provider in ALL_PROVIDERS:
            if provider.provider_id in ("nse_public", "dhan"):
                continue
            assert not provider.implemented, provider.provider_id
            assert not provider.verified, provider.provider_id

    def test_every_provider_has_documentation_to_check_against(self):
        for provider in ALL_PROVIDERS:
            assert provider.docs_url, provider.provider_id

    def test_ids_are_unique(self):
        ids = [p.provider_id for p in ALL_PROVIDERS]
        assert len(ids) == len(set(ids))
        assert set(ids) == set(PROVIDERS_BY_ID)

    def test_an_unknown_provider_raises_with_the_known_list(self):
        with pytest.raises(KeyError, match="Unknown provider"):
            get_provider("robinhood")

    def test_a_known_provider_resolves(self):
        assert get_provider("zerodha_kite") is ZERODHA_KITE


class TestCapabilityQueries:
    def test_only_dhan_can_serve_bars(self):
        """NSE's history endpoint blocks automated clients, so bars were the
        gap that motivated the Dhan adapter."""
        assert [p.provider_id for p in providers_serving(Capability.INDEX_BARS)] == [
            "dhan"
        ]

    def test_no_verified_provider_can_serve_bars_yet(self):
        """The stricter question, and the one that matters before relying on
        the number: Dhan's adapter exists but its mapping is unproven, so
        nothing verified serves bars."""
        assert providers_serving(Capability.INDEX_BARS, verified_only=True) == ()

    def test_only_dhan_can_trade(self):
        assert [p.provider_id for p in providers_serving(Capability.ORDER_PLACEMENT)] == [
            "dhan"
        ]

    def test_nse_serves_the_chain(self):
        served = providers_serving(Capability.OPTION_CHAIN, Capability.INDIA_VIX)
        assert [p.provider_id for p in served] == ["nse_public"]

    def test_documented_capabilities_do_not_satisfy_a_query(self):
        """Kite's documentation lists historical candles. The question this
        query answers is "who can serve me bars right now", and a documented
        capability with no adapter is not an answer to it."""
        assert Capability.INDEX_BARS in ZERODHA_KITE.capabilities
        assert ZERODHA_KITE not in providers_serving(Capability.INDEX_BARS)

    def test_nse_and_dhan_together_cover_analysis_and_trading(self):
        """The composition the whole capability split was designed for: NSE's
        chain and VIX, Dhan's bars and account."""
        from index_option_brain.data.providers import DHAN

        assert (
            missing_capabilities(
                NSE_PUBLIC_DESCRIPTOR, DHAN, required=REQUIRED_FOR_ANALYSIS
            )
            == frozenset()
        )
        assert (
            missing_capabilities(
                NSE_PUBLIC_DESCRIPTOR, DHAN, required=REQUIRED_FOR_TRADING
            )
            == frozenset()
        )

    def test_nse_alone_cannot_support_a_full_analysis(self):
        missing = missing_capabilities(
            NSE_PUBLIC_DESCRIPTOR, required=REQUIRED_FOR_ANALYSIS
        )
        assert missing == frozenset({Capability.INDEX_BARS})

    def test_nse_plus_a_broker_covers_the_analysis_requirements(self):
        """The composition the capability-split interfaces were designed for:
        one provider's gap filled by another's coverage."""
        assert (
            missing_capabilities(
                NSE_PUBLIC_DESCRIPTOR, ZERODHA_KITE, required=REQUIRED_FOR_ANALYSIS
            )
            == frozenset()
        )

    def test_breadth_is_still_uncovered_by_either(self):
        """Worth stating explicitly: constituent quotes are the one thing
        neither NSE public nor a broker's index API supplies here, so the
        Constituent brain runs degraded until something serves them."""
        assert missing_capabilities(
            NSE_PUBLIC_DESCRIPTOR,
            ZERODHA_KITE,
            required=frozenset({Capability.CONSTITUENT_QUOTES}),
        ) == frozenset({Capability.CONSTITUENT_QUOTES})

    def test_no_data_provider_can_trade(self):
        for provider in ALL_PROVIDERS:
            if provider.kind is ProviderKind.DATA:
                assert not provider.can_trade, provider.provider_id
                assert provider.missing(*REQUIRED_FOR_TRADING)


class TestDescriptorBehaviour:
    def test_supports_requires_all_capabilities(self):
        assert NSE_PUBLIC_DESCRIPTOR.supports(
            Capability.OPTION_CHAIN, Capability.INDIA_VIX
        )
        assert not NSE_PUBLIC_DESCRIPTOR.supports(
            Capability.OPTION_CHAIN, Capability.INDEX_BARS
        )

    def test_missing_names_the_gap(self):
        assert NSE_PUBLIC_DESCRIPTOR.missing(
            Capability.OPTION_CHAIN, Capability.INDEX_BARS
        ) == frozenset({Capability.INDEX_BARS})

    def test_capabilities_split_into_data_and_trading(self):
        assert NSE_PUBLIC_DESCRIPTOR.data_capabilities
        assert NSE_PUBLIC_DESCRIPTOR.trading_capabilities == frozenset()
        assert ZERODHA_KITE.trading_capabilities

    def test_the_two_capability_groups_do_not_overlap(self):
        assert DATA_CAPABILITIES & TRADING_CAPABILITIES == frozenset()

    def test_a_descriptor_is_immutable(self):
        with pytest.raises(ValidationError):
            NSE_PUBLIC_DESCRIPTOR.implemented = True  # type: ignore[misc]


class TestCredentialFields:
    def test_a_public_provider_needs_no_credentials(self):
        assert NSE_PUBLIC_DESCRIPTOR.auth is AuthMethod.NONE
        assert NSE_PUBLIC_DESCRIPTOR.credential_fields == ()

    def test_every_credentialed_provider_declares_its_fields(self):
        """The console renders these, so a provider with an auth method and no
        declared fields would show a connect form with nothing in it."""
        for provider in ALL_PROVIDERS:
            if provider.auth is AuthMethod.NONE:
                continue
            assert provider.credential_fields, provider.provider_id

    def test_secrets_are_marked_as_secret(self):
        secret_names = {"api_secret", "password", "totp_secret", "pin", "access_token"}
        for provider in ALL_PROVIDERS:
            for field in provider.credential_fields:
                if field.name in secret_names:
                    assert field.secret, f"{provider.provider_id}.{field.name}"

    def test_identifiers_are_not_marked_as_secret(self):
        """An API key or client ID has to be visible for an operator to check
        they pasted the right one."""
        for provider in ALL_PROVIDERS:
            for field in provider.credential_fields:
                if field.name in {"api_key", "client_id"}:
                    assert not field.secret, f"{provider.provider_id}.{field.name}"

    def test_a_totp_field_explains_what_to_paste(self):
        """Pasting the six-digit code instead of the seed is the obvious
        mistake, and it fails a minute later rather than immediately."""
        totp = next(
            field
            for field in get_provider("angel_one").credential_fields
            if field.name == "totp_secret"
        )
        assert "not the six-digit" in totp.help


class TestProviderHealth:
    def test_an_unconfigured_provider_reports_nothing_measured(self):
        """A console showing 0 ms latency for a provider it has never called
        is lying. Every field is either observed or None."""
        health = ProviderHealth(provider_id="nse_public")
        assert health.state is ProviderConnectionState.NOT_CONFIGURED
        assert health.latency_ms is None
        assert health.last_success_at is None
        assert health.verified_capabilities == frozenset()
        assert not health.is_usable

    def test_not_configured_is_distinct_from_disconnected(self):
        """"You never set this up" and "this was working and dropped" call
        for different operator actions."""
        assert (
            ProviderConnectionState.NOT_CONFIGURED
            is not ProviderConnectionState.DISCONNECTED
        )

    def test_degraded_is_still_usable(self):
        """Reachable but not serving everything it declared — which is
        precisely NSE's normal state, and it is worth trading analysis on."""
        health = ProviderHealth(
            provider_id="nse_public",
            state=ProviderConnectionState.DEGRADED,
            verified_capabilities=frozenset({Capability.OPTION_CHAIN}),
        )
        assert health.is_usable

    def test_failed_is_not_usable(self):
        health = ProviderHealth(
            provider_id="nse_public",
            state=ProviderConnectionState.FAILED,
            last_error="HTTP 403",
        )
        assert not health.is_usable

    def test_verified_capabilities_are_separate_from_declared_ones(self):
        """Declared is what the descriptor claims; verified is what a call
        actually returned. Only the second is evidence the feed works."""
        health = ProviderHealth(
            provider_id="nse_public",
            state=ProviderConnectionState.CONNECTED,
            verified_capabilities=frozenset({Capability.INDEX_QUOTE}),
        )
        assert health.verified_capabilities < NSE_PUBLIC_DESCRIPTOR.capabilities


class TestNsePublicDescriptorMatchesReality:
    """The descriptor is what the console trusts, so it has to match what the
    adapter actually does. Each absence below was probed, not assumed."""

    @pytest.mark.parametrize(
        "capability",
        [
            Capability.INDEX_QUOTE,
            Capability.EXPIRY_LIST,
            Capability.OPTION_CHAIN,
            Capability.INDIA_VIX,
        ],
    )
    def test_it_claims_what_the_endpoints_answered(self, capability):
        assert capability in NSE_PUBLIC_DESCRIPTOR.capabilities

    @pytest.mark.parametrize(
        ("capability", "reason"),
        [
            (Capability.INDEX_BARS, "history endpoint blocks automated clients"),
            (Capability.CONSTITUENT_QUOTES, "equity-stockIndices returns 404"),
            (Capability.OPTION_GREEKS, "only IV is published"),
            (Capability.ACCOUNT_SNAPSHOT, "not a broker"),
            (Capability.ORDER_PLACEMENT, "not a broker"),
            (Capability.STREAMING_QUOTES, "polled, not streamed"),
        ],
    )
    def test_it_does_not_claim_what_it_cannot_do(self, capability, reason):
        assert capability not in NSE_PUBLIC_DESCRIPTOR.capabilities, reason

    def test_its_notes_warn_about_the_missing_greeks(self):
        joined = " ".join(NSE_PUBLIC_DESCRIPTOR.notes)
        assert "No greeks" in joined
        assert "computed" in joined

    def test_its_notes_warn_about_the_missing_history(self):
        joined = " ".join(NSE_PUBLIC_DESCRIPTOR.notes)
        assert "No historical bars" in joined
