"""Provider capability descriptors (spec §2, §29).

The adapter interfaces are split by capability because Indian providers cover
different ground: NSE's public API serves the index, the full option chain and
India VIX but no historical bars and no account; a broker serves bars, account
and order placement but often a thinner chain. The system is meant to compose
one working data layer out of several partial providers.

That only works if capability is *data* rather than folklore. A descriptor
states what a provider serves, what credentials it needs, and — critically —
whether an adapter for it exists yet. The operations console renders these
descriptors directly, so a provider that is planned but unimplemented shows as
planned instead of appearing in a table that implies it works.

`implemented=False` is therefore load-bearing, not documentation: it is the
difference between an honest roadmap entry and a control that pretends to
connect to something.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ProviderKind(StrEnum):
    """What a provider is for.

    A DATA provider can never place an order, and the console must not offer
    one as an execution route.
    """

    DATA = "DATA"
    BROKER = "BROKER"
    DATA_AND_BROKER = "DATA_AND_BROKER"


class Capability(StrEnum):
    """One unit of provider coverage.

    Named after the adapter method it satisfies, so a missing capability maps
    to a specific call that will raise rather than to a vague shortfall.
    """

    INDEX_QUOTE = "INDEX_QUOTE"
    INDEX_BARS = "INDEX_BARS"
    CONSTITUENT_LIST = "CONSTITUENT_LIST"
    CONSTITUENT_QUOTES = "CONSTITUENT_QUOTES"
    EXPIRY_LIST = "EXPIRY_LIST"
    OPTION_CHAIN = "OPTION_CHAIN"
    OPTION_GREEKS = "OPTION_GREEKS"
    """Greeks published by the provider itself.

    Absent on every Indian source found so far, which is why the system
    computes them from IV in `analytics.pricing` rather than requiring them.
    """
    INDIA_VIX = "INDIA_VIX"
    STREAMING_QUOTES = "STREAMING_QUOTES"
    ACCOUNT_SNAPSHOT = "ACCOUNT_SNAPSHOT"
    MARGIN_CALCULATOR = "MARGIN_CALCULATOR"
    """A live SPAN + exposure margin quote for a basket.

    Worth distinguishing from ACCOUNT_SNAPSHOT: without it, margin has to be
    estimated locally, and an estimate that is too low is how a sized trade
    gets rejected by the broker after risk approved it.
    """
    ORDER_PLACEMENT = "ORDER_PLACEMENT"
    ORDER_MODIFICATION = "ORDER_MODIFICATION"
    ORDER_STREAMING = "ORDER_STREAMING"
    POSITION_BOOK = "POSITION_BOOK"
    BASKET_ORDERS = "BASKET_ORDERS"
    """Multi-leg submission in one request.

    A multi-leg structure sent leg-by-leg can be left half-filled, which turns
    a defined-risk spread into a naked short. Its absence is a real execution
    constraint, not a convenience gap.
    """


DATA_CAPABILITIES = frozenset(
    {
        Capability.INDEX_QUOTE,
        Capability.INDEX_BARS,
        Capability.CONSTITUENT_LIST,
        Capability.CONSTITUENT_QUOTES,
        Capability.EXPIRY_LIST,
        Capability.OPTION_CHAIN,
        Capability.OPTION_GREEKS,
        Capability.INDIA_VIX,
        Capability.STREAMING_QUOTES,
    }
)

TRADING_CAPABILITIES = frozenset(
    {
        Capability.ACCOUNT_SNAPSHOT,
        Capability.MARGIN_CALCULATOR,
        Capability.ORDER_PLACEMENT,
        Capability.ORDER_MODIFICATION,
        Capability.ORDER_STREAMING,
        Capability.POSITION_BOOK,
        Capability.BASKET_ORDERS,
    }
)


class AuthMethod(StrEnum):
    NONE = "NONE"
    """No credentials. Public endpoints only, and rate-limited accordingly."""

    API_KEY_SECRET = "API_KEY_SECRET"
    OAUTH_REQUEST_TOKEN = "OAUTH_REQUEST_TOKEN"
    """A browser login returns a request token that is exchanged for a session.

    Cannot be automated headlessly without storing the trading password, so a
    daily manual step is part of the operating procedure. The console has to
    show that rather than implying a connection persists forever.
    """
    OAUTH_TOTP = "OAUTH_TOTP"
    """Login plus a time-based one-time code, automatable from a TOTP secret."""


class CredentialField(BaseModel):
    """One input the console must collect to connect a provider."""

    model_config = ConfigDict(frozen=True)

    name: str
    label: str
    secret: bool = True
    """Secrets are write-only in the console and never echoed back."""
    required: bool = True
    help: str = ""


class ProviderDescriptor(BaseModel):
    """Everything the system and the console need to know about a provider."""

    model_config = ConfigDict(frozen=True)

    provider_id: str
    display_name: str
    kind: ProviderKind
    auth: AuthMethod
    capabilities: frozenset[Capability]
    credential_fields: tuple[CredentialField, ...] = ()
    implemented: bool = False
    """Whether an adapter exists in this codebase.

    False means the descriptor is a roadmap entry. The console must render it
    as unavailable and must not offer a connect action for it.
    """
    verified: bool = False
    """Whether the adapter's field mapping was checked against real responses.

    Distinct from `implemented`, and the distinction is not pedantry. An
    adapter can exist, compile, and be entirely wrong about which field holds
    the bid — NSE documents `bidprice` and always sends null, putting the real
    top of book in `buyPrice1`, so an adapter written from the documentation
    alone would produce a chain with no bid anywhere and no error to show for
    it.

    So: `implemented` says code exists, `verified` says the code was proven
    against a payload the provider actually sent, and
    `ProviderHealth.verified_capabilities` says which calls worked just now.
    Three different questions.
    """
    docs_url: str | None = None
    notes: tuple[str, ...] = ()
    """Operating caveats worth showing an operator before they rely on it."""

    def supports(self, *capabilities: Capability) -> bool:
        return all(capability in self.capabilities for capability in capabilities)

    def missing(self, *capabilities: Capability) -> frozenset[Capability]:
        return frozenset(capabilities) - self.capabilities

    @property
    def can_trade(self) -> bool:
        return Capability.ORDER_PLACEMENT in self.capabilities

    @property
    def data_capabilities(self) -> frozenset[Capability]:
        return self.capabilities & DATA_CAPABILITIES

    @property
    def trading_capabilities(self) -> frozenset[Capability]:
        return self.capabilities & TRADING_CAPABILITIES


class ProviderConnectionState(StrEnum):
    """How a configured provider is actually doing right now.

    NOT_CONFIGURED is the honest default and must be distinguishable from
    DISCONNECTED: "you never set this up" and "this was working and has
    dropped" call for different operator actions.
    """

    NOT_CONFIGURED = "NOT_CONFIGURED"
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    DEGRADED = "DEGRADED"
    """Reachable, but not serving everything it declared."""
    FAILED = "FAILED"


class ProviderHealth(BaseModel):
    """A live health reading for one provider.

    Every field is either observed or None. There are no defaults standing in
    for measurements: a console that shows a latency of 0 ms for a provider it
    has never called is lying, and an operator deciding whether to trade on a
    feed needs to be able to tell "healthy" from "unknown".
    """

    model_config = ConfigDict(frozen=True)

    provider_id: str
    state: ProviderConnectionState = ProviderConnectionState.NOT_CONFIGURED
    checked_at: object | None = None
    latency_ms: float | None = None
    last_success_at: object | None = None
    last_error: str | None = None
    verified_capabilities: frozenset[Capability] = Field(default_factory=frozenset)
    """Capabilities proven by an actual successful call, not merely declared."""

    @property
    def is_usable(self) -> bool:
        return self.state in (
            ProviderConnectionState.CONNECTED,
            ProviderConnectionState.DEGRADED,
        )
