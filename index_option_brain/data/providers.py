"""The registry of providers the system knows about (spec §2).

One place that answers "what can I connect to, and what would it give me".
The operations console renders these descriptors directly, which is what keeps
the connection UI honest: a provider with no adapter shows as having no
adapter instead of appearing in a list that implies it works.

On the accuracy of what is claimed here
---------------------------------------
Exactly one entry has `implemented=True` — NSE's public API — and every
capability on it was probed against the live endpoint. Its four capabilities
are the ones that answered; bars, breadth, greeks, account and orders are
absent because they were tried and did not work.

Every other entry is a **roadmap entry**. Its capability list is read off the
provider's published API documentation and has not been verified against an
adapter in this codebase, and each one says so in its notes. That distinction
is the point of the registry: a documented capability is a claim, a probed one
is a fact, and a trading system must not treat them alike. When an adapter is
built, its capabilities get corrected to what the provider actually serves —
which, on the evidence of NSE, will not be the same list.

Lot sizes, margin rules and API surfaces all change by circular and release
notes. Treat this file as needing review, not as a reference.
"""

from __future__ import annotations

from index_option_brain.contracts.provider import (
    AuthMethod,
    Capability,
    CredentialField,
    ProviderDescriptor,
    ProviderKind,
)
from index_option_brain.data.adapters.nse_public import NSE_PUBLIC_DESCRIPTOR

_UNVERIFIED = (
    "Capabilities are read from the provider's published API documentation "
    "and have not been verified against an adapter. No adapter exists yet."
)

# Credential shapes that recur, defined once so the console renders the same
# field for the same concept across providers.
_API_KEY = CredentialField(
    name="api_key",
    label="API key",
    secret=False,
    help="Issued when you create an app in the provider's developer console.",
)
_API_SECRET = CredentialField(name="api_secret", label="API secret")
_CLIENT_ID = CredentialField(
    name="client_id", label="Client / user ID", secret=False
)
_TOTP_SECRET = CredentialField(
    name="totp_secret",
    label="TOTP secret",
    help=(
        "The base32 seed behind your authenticator app, not the six-digit "
        "code. It lets the session be renewed without a manual login."
    ),
)
_PASSWORD = CredentialField(name="password", label="Login password")
_PIN = CredentialField(name="pin", label="Trading PIN / MPIN")

# What a full-service broker API is generally expected to cover. Named so the
# roadmap entries below read as "documented as offering this", not as a
# measured capability set.
_DOCUMENTED_BROKER_CAPABILITIES = frozenset(
    {
        Capability.INDEX_QUOTE,
        Capability.INDEX_BARS,
        Capability.EXPIRY_LIST,
        Capability.OPTION_CHAIN,
        Capability.STREAMING_QUOTES,
        Capability.ACCOUNT_SNAPSHOT,
        Capability.ORDER_PLACEMENT,
        Capability.ORDER_MODIFICATION,
        Capability.POSITION_BOOK,
    }
)


def _broker(
    provider_id: str,
    display_name: str,
    *,
    auth: AuthMethod,
    credentials: tuple[CredentialField, ...],
    docs_url: str,
    extra: frozenset[Capability] = frozenset(),
    notes: tuple[str, ...] = (),
) -> ProviderDescriptor:
    return ProviderDescriptor(
        provider_id=provider_id,
        display_name=display_name,
        kind=ProviderKind.DATA_AND_BROKER,
        auth=auth,
        capabilities=_DOCUMENTED_BROKER_CAPABILITIES | extra,
        credential_fields=credentials,
        implemented=False,
        docs_url=docs_url,
        notes=(_UNVERIFIED, *notes),
    )


ZERODHA_KITE = _broker(
    "zerodha_kite",
    "Zerodha Kite Connect",
    auth=AuthMethod.OAUTH_REQUEST_TOKEN,
    credentials=(_API_KEY, _API_SECRET),
    docs_url="https://kite.trade/docs/connect/v3/",
    extra=frozenset({Capability.ORDER_STREAMING, Capability.MARGIN_CALCULATOR}),
    notes=(
        (
            "The login flow returns a request token in a browser redirect, "
            "which is exchanged for a day-long access token. It cannot be "
            "automated without storing the trading password, so a manual "
            "login is part of the daily operating procedure."
        ),
        "Historical candles are a separately priced subscription.",
    ),
)

UPSTOX = _broker(
    "upstox",
    "Upstox",
    auth=AuthMethod.OAUTH_REQUEST_TOKEN,
    credentials=(_API_KEY, _API_SECRET),
    docs_url="https://upstox.com/developer/api-documentation/",
    extra=frozenset({Capability.ORDER_STREAMING, Capability.MARGIN_CALCULATOR}),
    notes=("Access tokens expire daily and require a browser login to renew.",),
)

ANGEL_ONE = _broker(
    "angel_one",
    "Angel One SmartAPI",
    auth=AuthMethod.OAUTH_TOTP,
    credentials=(_API_KEY, _CLIENT_ID, _PASSWORD, _TOTP_SECRET),
    docs_url="https://smartapi.angelbroking.com/docs",
    extra=frozenset({Capability.MARGIN_CALCULATOR}),
    notes=(
        (
            "TOTP login means the session can be renewed without a human, "
            "which matters for an unattended process."
        ),
    ),
)

DHAN = _broker(
    "dhan",
    "Dhan",
    auth=AuthMethod.API_KEY_SECRET,
    credentials=(
        _CLIENT_ID,
        CredentialField(
            name="access_token",
            label="Access token",
            help="Generated in the Dhan web console; longer-lived than a daily OAuth token.",
        ),
    ),
    docs_url="https://dhanhq.co/docs/v2/",
    extra=frozenset({Capability.MARGIN_CALCULATOR}),
    notes=(
        (
            "A long-lived token avoids the daily browser login the OAuth "
            "brokers require, which suits an unattended process."
        ),
    ),
)

FYERS = _broker(
    "fyers",
    "Fyers",
    auth=AuthMethod.OAUTH_TOTP,
    credentials=(_API_KEY, _API_SECRET, _CLIENT_ID, _TOTP_SECRET, _PIN),
    docs_url="https://myapi.fyers.in/docsv3",
)

ICICI_BREEZE = _broker(
    "icici_breeze",
    "ICICI Direct Breeze",
    auth=AuthMethod.API_KEY_SECRET,
    credentials=(_API_KEY, _API_SECRET),
    docs_url="https://api.icicidirect.com/apiuser/home",
    notes=("A session token has to be generated from a browser login each day.",),
)

KOTAK_NEO = _broker(
    "kotak_neo",
    "Kotak Neo",
    auth=AuthMethod.OAUTH_TOTP,
    credentials=(_API_KEY, _API_SECRET, _CLIENT_ID, _PASSWORD, _TOTP_SECRET),
    docs_url="https://documenter.getpostman.com/view/21534797/UzBnqmpD",
)

FIVE_PAISA = _broker(
    "five_paisa",
    "5paisa",
    auth=AuthMethod.OAUTH_TOTP,
    credentials=(_API_KEY, _API_SECRET, _CLIENT_ID, _TOTP_SECRET, _PIN),
    docs_url="https://www.5paisa.com/developerapi/",
)

FINVASIA_SHOONYA = _broker(
    "finvasia_shoonya",
    "Finvasia Shoonya",
    auth=AuthMethod.OAUTH_TOTP,
    credentials=(_CLIENT_ID, _PASSWORD, _TOTP_SECRET),
    docs_url="https://shoonya.com/api-documentation/",
    notes=("The API is offered without a subscription fee.",),
)

ALICE_BLUE = _broker(
    "alice_blue",
    "Alice Blue ANT",
    auth=AuthMethod.API_KEY_SECRET,
    credentials=(_API_KEY, _CLIENT_ID),
    docs_url="https://v2api.aliceblueonline.com/introduction",
)


ALL_PROVIDERS: tuple[ProviderDescriptor, ...] = (
    NSE_PUBLIC_DESCRIPTOR,
    ZERODHA_KITE,
    UPSTOX,
    ANGEL_ONE,
    DHAN,
    FYERS,
    ICICI_BREEZE,
    KOTAK_NEO,
    FIVE_PAISA,
    FINVASIA_SHOONYA,
    ALICE_BLUE,
)

PROVIDERS_BY_ID: dict[str, ProviderDescriptor] = {
    provider.provider_id: provider for provider in ALL_PROVIDERS
}


def get_provider(provider_id: str) -> ProviderDescriptor:
    try:
        return PROVIDERS_BY_ID[provider_id]
    except KeyError:
        raise KeyError(
            f"Unknown provider {provider_id!r}. Known: {sorted(PROVIDERS_BY_ID)}"
        ) from None


def implemented_providers() -> tuple[ProviderDescriptor, ...]:
    """Providers with a working adapter — the only ones that can be connected."""
    return tuple(p for p in ALL_PROVIDERS if p.implemented)


def providers_serving(*capabilities: Capability) -> tuple[ProviderDescriptor, ...]:
    """Implemented providers covering all of `capabilities`.

    Restricted to implemented providers on purpose: the question this answers
    is "who can serve me bars right now", and a documented capability with no
    adapter behind it is not an answer to that.
    """
    return tuple(
        p for p in implemented_providers() if p.supports(*capabilities)
    )


def missing_capabilities(
    *providers: ProviderDescriptor, required: frozenset[Capability]
) -> frozenset[Capability]:
    """What a set of providers still cannot cover between them.

    The capability-split design exists so several partial providers compose
    into one working data layer, and this is the question that makes the
    composition checkable: NSE plus a broker covers the chain and the bars,
    but neither covers index breadth.
    """
    covered: frozenset[Capability] = frozenset()
    for provider in providers:
        covered |= provider.capabilities
    return required - covered


REQUIRED_FOR_ANALYSIS = frozenset(
    {
        Capability.INDEX_QUOTE,
        Capability.INDEX_BARS,
        Capability.EXPIRY_LIST,
        Capability.OPTION_CHAIN,
    }
)
"""The minimum for the brains to produce a full analysis.

India VIX and constituent breadth are not in this set: the volatility and
constituent brains degrade honestly without them. Bars are, because with no
bars there is no measured structure and the Regime Engine correctly refuses
to classify anything.
"""

REQUIRED_FOR_TRADING = frozenset(
    {
        Capability.ACCOUNT_SNAPSHOT,
        Capability.ORDER_PLACEMENT,
        Capability.POSITION_BOOK,
    }
)
"""The minimum to place and manage a trade. No data provider has these."""
