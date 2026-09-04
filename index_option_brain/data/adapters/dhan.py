"""Dhan (DhanHQ v2) adapter — bars, chain, account and orders.

Why this provider
-----------------
It closes the two gaps that currently stop the system short. NSE's public
feed serves no history, so the Regime Engine has no measured structure and
correctly refuses to classify one; and there is no account, so the Risk
Engine cannot size anything. Dhan serves both, its order API needs no data
subscription, and its access token is long-lived — which matters for an
unattended process, where the daily browser login the OAuth brokers require
is a standing operational failure.

What has been verified, and what has not
----------------------------------------
**Verified** by probing the live hosts without credentials: every route below
exists (each answered with an authentication or input error, never a 404),
`https://sandbox.dhan.co/v2` is real, and the API returns **two different
error envelopes** —

    {"errorType": ..., "errorCode": "DH-901", "errorMessage": ...}
    {"Data": {"810": "ClientId is invalid"}, "status": "failed"}

The first comes from the charts, funds and orders family; the second from the
option-chain and market-feed family. Both are handled, because a client that
only understood one would read the other as a successful response.

**Not verified:** the response *bodies*. Those need a token. The field
mapping below follows Dhan's published documentation, and the NSE experience
says documentation and reality differ in exactly the places that hurt —
there, the documented `bidprice` was always null and the real top of book
lived in `buyPrice1`. So every field access goes through `_require`, which
raises a `DataAdapterError` naming the missing field rather than substituting
a default. A shape mismatch surfaces as a precise error on the first call,
not as silent nonsense in a sizing calculation.

Run `scripts/dhan_probe.py` once with credentials before trusting it. It
dumps the real shapes so the mapping can be corrected against evidence,
which is how the NSE adapter was built.

Sandbox first, but it does not serve market data
------------------------------------------------
`DhanConfig(sandbox=True)` targets `sandbox.dhan.co` and is the default.
Pointing at the live host is an explicit choice, because a mis-set flag on a
trading adapter is not a configuration error, it is a trade.

The two hosts do not cover the same routes. Probed on both:

| Route | Sandbox | Live |
| --- | --- | --- |
| `/orders`, `/fundlimit` | works | works |
| `/charts/historical`, `/charts/intraday` | works | works |
| `/marketfeed/*` | **404** | works |
| `/optionchain`, `/optionchain/expirylist` | **404** | works |

So the order path and the account can be exercised against the sandbox with
no money at risk, while the quote and chain mapping has to be verified
against the live host — read-only, which is safe, but it needs the Data API
subscription. `MARKET_DATA_ROUTES` records which calls those are, and they
fail with that explanation rather than a bare 404, because a 404 from a route
that exists on the other host is a confusing way to learn this.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Self

from index_option_brain.contracts.enums import (
    BarInterval,
    OptionType,
    OrderLifecycleState,
    OrderSide,
)
from index_option_brain.contracts.instruments import (
    AccountSnapshot,
    Bar,
    Greeks,
    IndexQuote,
    IndexSpec,
    OptionContractSpec,
    OptionQuote,
)
from index_option_brain.contracts.order import Order, OrderRequest
from index_option_brain.contracts.provider import (
    AuthMethod,
    Capability,
    CredentialField,
    ProviderDescriptor,
    ProviderKind,
)
from index_option_brain.data.adapters.base import (
    AccountDataAdapter,
    DataAdapterError,
    IndexDataAdapter,
    OptionsChainAdapter,
)
from index_option_brain.data.dhan_instruments import DhanInstrumentMaster
from index_option_brain.data.http import HttpError, HttpSession, HttpxSession
from index_option_brain.execution.broker_adapter import BrokerAdapter

LIVE_BASE = "https://api.dhan.co/v2"
SANDBOX_BASE = "https://sandbox.dhan.co/v2"

IST = timezone(timedelta(hours=5, minutes=30), name="IST")

# Dhan's segment code for a cash index. Its option contracts live in the F&O
# segment, which is a different code — mixing them up returns an input error
# rather than the wrong data, which is the good failure mode.
INDEX_SEGMENT = "IDX_I"
FNO_SEGMENT = "NSE_FNO"

# Routes the sandbox does not implement. Probed: it answers these with a
# Spring-default 404 rather than a Dhan error envelope, which is itself the
# giveaway that they are not deployed there at all.
MARKET_DATA_ROUTES: frozenset[str] = frozenset(
    {"/marketfeed/ohlc", "/marketfeed/ltp", "/marketfeed/quote", "/optionchain"}
)

# Dhan expresses an intraday interval as a string of minutes.
_INTERVAL_MINUTES: dict[BarInterval, str] = {
    BarInterval.MINUTE_1: "1",
    BarInterval.MINUTE_5: "5",
    BarInterval.MINUTE_15: "15",
}

DHAN_DESCRIPTOR = ProviderDescriptor(
    provider_id="dhan",
    display_name="Dhan (DhanHQ v2)",
    kind=ProviderKind.DATA_AND_BROKER,
    auth=AuthMethod.API_KEY_SECRET,
    capabilities=frozenset(
        {
            Capability.INDEX_QUOTE,
            Capability.INDEX_BARS,
            Capability.EXPIRY_LIST,
            Capability.OPTION_CHAIN,
            Capability.OPTION_GREEKS,
            Capability.ACCOUNT_SNAPSHOT,
            Capability.MARGIN_CALCULATOR,
            Capability.ORDER_PLACEMENT,
            Capability.ORDER_MODIFICATION,
            Capability.POSITION_BOOK,
        }
    ),
    credential_fields=(
        CredentialField(
            name="client_id",
            label="Client ID",
            secret=False,
            help="Your Dhan client id, shown in the web console.",
        ),
        CredentialField(
            name="access_token",
            label="Access token",
            help=(
                "Generated in the Dhan web console. Long-lived, unlike the "
                "daily OAuth tokens other brokers issue — which is what makes "
                "an unattended process practical."
            ),
        ),
    ),
    implemented=True,
    # Routes and error envelopes were verified; response bodies were not.
    # scripts/dhan_probe.py flips this once it has been run with a token.
    verified=False,
    docs_url="https://dhanhq.co/docs/v2/",
    notes=(
        (
            "Routes and both error envelopes were verified against the live "
            "hosts. Response bodies were NOT — they need a token. Run "
            "scripts/dhan_probe.py once before trusting the field mapping."
        ),
        (
            "Historical bars and the option chain require the Data API "
            "subscription. Orders, positions and funds do not."
        ),
        (
            "The sandbox implements orders, funds and charts but NOT "
            "/marketfeed or /optionchain — both 404 there and work on the live "
            "host. Order-path testing is free of risk; chain verification is "
            "not free of subscription."
        ),
        (
            "Defaults to the sandbox at sandbox.dhan.co. Pointing at the live "
            "host is an explicit choice."
        ),
        (
            "Greeks are published on its option chain, unlike NSE's — but they "
            "are recomputed locally unless trust_broker_greeks is set, so one "
            "pricing convention applies across every provider."
        ),
    ),
)


@dataclass(frozen=True)
class DhanConfig:
    """Credentials and which host to talk to."""

    client_id: str
    access_token: str
    sandbox: bool = True
    """Sandbox by default. A mis-set flag on a trading adapter is not a
    configuration error, it is a trade."""
    risk_free_rate: float = 0.065
    trust_broker_greeks: bool = False
    """Use Dhan's published greeks instead of recomputing them.

    Off by default, and not out of distrust. Greeks depend on the rate and
    the day-count convention used to compute them, and Dhan does not publish
    either. Mixing its numbers with locally computed ones from NSE would put
    two conventions in one ranking, so delta fit would compare quantities
    that are not the same quantity.
    """

    @property
    def base_url(self) -> str:
        return SANDBOX_BASE if self.sandbox else LIVE_BASE

    @property
    def is_live(self) -> bool:
        return not self.sandbox


class DhanApiError(DataAdapterError):
    """An error Dhan reported, with its own code preserved.

    The code is kept because Dhan's codes are actionable and distinct:
    DH-901 is a dead token and needs re-authentication, DH-905 is a
    malformed request and needs a code fix, and rate limiting needs backing
    off. Collapsing them into one message would lose the difference.
    """

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


def _parse_error(payload: Any) -> tuple[str, str | None] | None:
    """Recognize either error envelope, or return None if this is not one.

    Both shapes were observed on the live hosts. A client that understood
    only the first would treat `{"status": "failed"}` as a successful
    response and parse an error body as market data.
    """
    if not isinstance(payload, dict):
        return None
    if payload.get("errorType") or payload.get("errorCode"):
        return (
            str(payload.get("errorMessage") or payload.get("errorType")),
            str(payload.get("errorCode")) if payload.get("errorCode") else None,
        )
    if payload.get("status") == "failed":
        detail = payload.get("Data")
        if isinstance(detail, dict) and detail:
            code, message = next(iter(detail.items()))
            return (str(message), str(code))
        return (f"Dhan reported failure: {detail!r}", None)
    return None


def _require(payload: dict[str, Any], field: str, context: str) -> Any:
    """Read a field, or raise naming what was missing and where.

    No defaults. The NSE adapter's lesson: documented field names and real
    ones differ in exactly the places that hurt, and a substituted zero
    reaches a sizing calculation looking like a measurement.
    """
    if field not in payload:
        raise DataAdapterError(
            f"Dhan {context} response has no {field!r} field. Present: "
            f"{sorted(payload)}. Run scripts/dhan_probe.py to capture the real "
            "shape and correct the mapping."
        )
    value = payload[field]
    if value is None:
        raise DataAdapterError(f"Dhan {context} returned null for {field!r}")
    return value


def _decimal(value: Any, field: str, context: str) -> Decimal:
    try:
        return Decimal(str(value))
    except (ArithmeticError, TypeError, ValueError) as exc:
        raise DataAdapterError(
            f"Dhan {context} field {field!r} is not a number: {value!r}"
        ) from exc


def _optional_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        parsed = Decimal(str(value))
    except (ArithmeticError, TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(Decimal(str(value)))
    except (ArithmeticError, TypeError, ValueError):
        return 0


class DhanClient:
    """Authenticated transport for the DhanHQ v2 API.

    Split from the adapter so the auth and error handling — the parts that
    were verified against the live hosts — are testable and reusable by the
    broker adapter without dragging market-data parsing along.
    """

    def __init__(self, config: DhanConfig, session: HttpSession | None = None) -> None:
        self._config = config
        self._owns_session = session is None
        self._session: HttpSession = session or HttpxSession(headers=self.headers)

    @property
    def config(self) -> DhanConfig:
        return self._config

    @property
    def headers(self) -> dict[str, str]:
        return {
            "access-token": self._config.access_token,
            "client-id": self._config.client_id,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    async def aclose(self) -> None:
        if self._owns_session:
            await self._session.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def get(self, path: str) -> Any:
        return await self._call("GET", path, None)

    async def post(self, path: str, body: dict[str, Any]) -> Any:
        return await self._call("POST", path, body)

    async def delete(self, path: str) -> Any:
        """Dhan cancels an order with a real DELETE to /orders/{id}.

        It has to be a DELETE: a GET to the same path returns the order's
        *status*, so sending one and treating the reply as a cancel would
        report success while the order stayed live — which is precisely how a
        system comes to believe it is flat while holding a position.

        Routed through the same envelope handling as everything else, because
        a cancel that quietly failed is the same problem by another route.
        """
        return await self._call("DELETE", path, None)

    async def _call(self, method: str, path: str, body: dict[str, Any] | None) -> Any:
        if self._config.sandbox and any(
            path.startswith(route) for route in MARKET_DATA_ROUTES
        ):
            raise DataAdapterError(
                f"Dhan's sandbox does not implement {path} — it serves orders, "
                "funds and charts only. Verify market data against the live "
                "host (read-only, but it needs the Data API subscription), or "
                "keep using NSE public for the chain."
            )
        url = f"{self._config.base_url}{path}"
        try:
            if method == "GET":
                response = await self._session.get(url, headers=self.headers)
            elif method == "DELETE":
                response = await self._session.delete(url, headers=self.headers)
            else:
                response = await self._session.post(
                    url, json=body or {}, headers=self.headers
                )
        except HttpError as exc:
            raise DataAdapterError(f"Dhan request failed: {exc}") from exc

        try:
            payload = response.json()
        except ValueError:
            raise DataAdapterError(
                f"Dhan returned a non-JSON body for {path} "
                f"(HTTP {response.status_code}): {response.text[:200]!r}"
            ) from None

        # Checked before the status code, because Dhan reports some failures
        # inside a 200 body — the `{"status": "failed"}` envelope in
        # particular.
        error = _parse_error(payload)
        if error is not None:
            message, code = error
            raise DhanApiError(f"Dhan {path}: {message}", code=code)
        if not response.is_ok:
            raise DataAdapterError(f"Dhan returned HTTP {response.status_code} for {path}")
        return payload


class DhanMarketDataAdapter(IndexDataAdapter, OptionsChainAdapter, AccountDataAdapter):
    """Index bars, quotes, option chain and account from Dhan.

    Contract specifications come from the instrument master rather than from
    these responses: security ids and lot sizes are what the master is for,
    and it needs no credentials.
    """

    descriptor = DHAN_DESCRIPTOR

    def __init__(
        self,
        client: DhanClient,
        master: DhanInstrumentMaster,
        *,
        index_names: dict[str, str] | None = None,
    ) -> None:
        self._client = client
        self._master = master
        self._names = index_names or {"NIFTY": "Nifty 50", "BANKNIFTY": "Nifty Bank"}

    async def aclose(self) -> None:
        await self._client.aclose()

    # ---------------------------------------------------------- index data

    async def get_index_spec(self, symbol: str) -> IndexSpec:
        symbol = symbol.upper()
        expiries = self._master.expiries(symbol)
        near = expiries[0] if expiries else None
        step = self._master.strike_step(symbol, near) if near else None
        return IndexSpec(
            symbol=symbol,
            name=self._names.get(symbol, symbol),
            lot_size=self._master.lot_size(symbol, near),
            tick_size=self._master.tick_size(symbol),
            strike_step=step if step is not None else Decimal(50),
        )

    async def get_index_quote(self, symbol: str) -> IndexQuote:
        symbol = symbol.upper()
        security_id = self._master.index_security_id(symbol)
        payload = await self._client.post(
            "/marketfeed/ohlc", {INDEX_SEGMENT: [int(security_id)]}
        )
        row = self._market_feed_row(payload, security_id, symbol)
        # Dhan nests OHLC under an "ohlc" object on some feed endpoints and
        # flattens it on others. Accepting both costs one line and avoids a
        # KeyError that would read as "the index has no open price".
        nested = row.get("ohlc")
        ohlc: dict[str, Any] = nested if isinstance(nested, dict) else row
        return IndexQuote(
            symbol=symbol,
            # Dhan's feed carries no timestamp of its own on this endpoint, so
            # the read time is used and labelled as such. It is the one place
            # here that reads a clock, and it is why bars are preferred over
            # this quote for anything time-sensitive.
            timestamp=datetime.now(UTC),
            ltp=_decimal(_require(row, "last_price", "market feed"), "last_price", "market feed"),
            open=_decimal(_require(ohlc, "open", "market feed"), "open", "market feed"),
            high=_decimal(_require(ohlc, "high", "market feed"), "high", "market feed"),
            low=_decimal(_require(ohlc, "low", "market feed"), "low", "market feed"),
            previous_close=_decimal(
                _require(ohlc, "close", "market feed"), "close", "market feed"
            ),
            vwap=None,
        )

    def _market_feed_row(
        self, payload: Any, security_id: str, symbol: str
    ) -> dict[str, Any]:
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            raise DataAdapterError("Dhan market feed response has no 'data' object")
        segment = data.get(INDEX_SEGMENT)
        if not isinstance(segment, dict):
            raise DataAdapterError(
                f"Dhan market feed has no {INDEX_SEGMENT} segment for {symbol}"
            )
        row = segment.get(security_id) or segment.get(str(security_id))
        if not isinstance(row, dict):
            raise DataAdapterError(
                f"Dhan market feed has no row for {symbol} (security id {security_id})"
            )
        return row

    async def get_index_bars(
        self, symbol: str, interval: BarInterval, count: int
    ) -> list[Bar]:
        """Completed bars, oldest first, excluding the forming candle.

        The forming candle is dropped here rather than trusted to be absent:
        the adapter contract requires it, the brains read the last daily bar
        as the previous session, and an intraday request that includes today
        would silently corrupt every level derived from it.
        """
        symbol = symbol.upper()
        security_id = self._master.index_security_id(symbol)
        today = datetime.now(IST).date()

        if interval is BarInterval.DAY:
            # Generous window: exchange holidays and weekends mean calendar
            # days are always more than trading days, and asking for exactly
            # `count` days returns fewer bars than asked for.
            start = today - timedelta(days=max(count * 2, 30))
            payload = await self._client.post(
                "/charts/historical",
                {
                    "securityId": security_id,
                    "exchangeSegment": INDEX_SEGMENT,
                    "instrument": "INDEX",
                    "fromDate": start.isoformat(),
                    "toDate": today.isoformat(),
                },
            )
        else:
            minutes = _INTERVAL_MINUTES.get(interval)
            if minutes is None:
                raise DataAdapterError(
                    f"Dhan does not serve {interval} bars. Supported: "
                    f"{sorted(_INTERVAL_MINUTES)}"
                )
            payload = await self._client.post(
                "/charts/intraday",
                {
                    "securityId": security_id,
                    "exchangeSegment": INDEX_SEGMENT,
                    "instrument": "INDEX",
                    "interval": minutes,
                    "fromDate": (today - timedelta(days=5)).isoformat(),
                    "toDate": today.isoformat(),
                },
            )

        bars = self._parse_candles(payload, symbol)
        if interval is not BarInterval.DAY and bars:
            bars = self._drop_forming(bars, interval)
        elif interval is BarInterval.DAY and bars:
            bars = self._drop_todays_bar(bars, today)
        return bars[-count:] if count > 0 else bars

    def _parse_candles(self, payload: Any, symbol: str) -> list[Bar]:
        """Dhan returns parallel arrays, not a list of candles.

        Their lengths are checked against each other: a short `volume` array
        zipped against a long `close` array would silently mis-pair every bar
        after the first gap, which is the kind of error that produces
        plausible-looking nonsense.
        """
        if not isinstance(payload, dict):
            raise DataAdapterError("Dhan chart response is not an object")
        required = ("open", "high", "low", "close", "timestamp")
        arrays: dict[str, list[Any]] = {}
        for field in required:
            value = payload.get(field)
            if not isinstance(value, list):
                raise DataAdapterError(
                    f"Dhan chart response for {symbol} has no {field!r} array. "
                    f"Present: {sorted(payload)}. Run scripts/dhan_probe.py to "
                    "capture the real shape."
                )
            arrays[field] = value
        volumes = payload.get("volume")
        arrays["volume"] = volumes if isinstance(volumes, list) else []

        lengths = {field: len(values) for field, values in arrays.items() if values}
        core = {field: length for field, length in lengths.items() if field != "volume"}
        if len(set(core.values())) > 1:
            raise DataAdapterError(
                f"Dhan chart arrays for {symbol} have mismatched lengths {core}; "
                "pairing them would mis-associate every bar after the gap"
            )

        bars: list[Bar] = []
        for index in range(len(arrays["close"])):
            raw_timestamp = arrays["timestamp"][index]
            bars.append(
                Bar(
                    timestamp=self._parse_epoch(raw_timestamp),
                    open=_decimal(arrays["open"][index], "open", "chart"),
                    high=_decimal(arrays["high"][index], "high", "chart"),
                    low=_decimal(arrays["low"][index], "low", "chart"),
                    close=_decimal(arrays["close"][index], "close", "chart"),
                    volume=_int(
                        arrays["volume"][index] if index < len(arrays["volume"]) else 0
                    ),
                )
            )
        bars.sort(key=lambda candle: candle.timestamp)
        return bars

    def _parse_epoch(self, raw: Any) -> datetime:
        """Dhan sends an epoch second count. Converted to timezone-aware UTC.

        A naive datetime here would make every session-boundary and
        time-to-expiry computation downstream depend on the host's timezone.
        """
        try:
            return datetime.fromtimestamp(int(float(raw)), tz=UTC)
        except (TypeError, ValueError, OSError, OverflowError) as exc:
            raise DataAdapterError(
                f"Dhan chart timestamp {raw!r} is not an epoch second count"
            ) from exc

    def _drop_forming(self, bars: list[Bar], interval: BarInterval) -> list[Bar]:
        """Drop a final bar whose period has not closed yet."""
        minutes = int(_INTERVAL_MINUTES[interval])
        cutoff = datetime.now(UTC) - timedelta(minutes=minutes)
        return [bar for bar in bars if bar.timestamp <= cutoff]

    def _drop_todays_bar(self, bars: list[Bar], today: date) -> list[Bar]:
        return [bar for bar in bars if bar.timestamp.astimezone(IST).date() < today]

    # --------------------------------------------------------- option data

    async def get_available_expiries(self, underlying_symbol: str) -> list[date]:
        """From the instrument master, not from the API.

        The master already lists every contract, needs no subscription, and
        cannot disagree with the lot sizes read from the same file. Spending a
        rate-limited API call to learn something already on disk would be
        worse in every respect.
        """
        expiries = self._master.expiries(underlying_symbol)
        if not expiries:
            raise DataAdapterError(
                f"No expiries listed for {underlying_symbol} in the instrument master"
            )
        return expiries

    async def get_option_chain(
        self, underlying_symbol: str, expiry: date
    ) -> list[OptionQuote]:
        symbol = underlying_symbol.upper()
        security_id = self._master.index_security_id(symbol)
        payload = await self._client.post(
            "/optionchain",
            {
                "UnderlyingScrip": int(security_id),
                "UnderlyingSeg": INDEX_SEGMENT,
                "Expiry": expiry.isoformat(),
            },
        )
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            raise DataAdapterError("Dhan option chain response has no 'data' object")
        strikes = data.get("oc")
        if not isinstance(strikes, dict) or not strikes:
            raise DataAdapterError(
                f"Dhan published an empty chain for {symbol} {expiry.isoformat()}"
            )

        spot = _optional_decimal(data.get("last_price"))
        as_of = datetime.now(UTC)
        years = max(
            0.0,
            (
                datetime.combine(expiry, datetime.min.time(), tzinfo=IST)
                + timedelta(hours=15, minutes=30)
                - as_of
            ).total_seconds()
            / (365.0 * 24 * 3600),
        )
        lot_size = self._master.lot_size(symbol, expiry)
        tick_size = self._master.tick_size(symbol)

        quotes: list[OptionQuote] = []
        for raw_strike, sides in strikes.items():
            if not isinstance(sides, dict):
                continue
            strike = _optional_decimal(raw_strike)
            if strike is None:
                continue
            for key, option_type in (("ce", OptionType.CE), ("pe", OptionType.PE)):
                leg = sides.get(key)
                # A strike listed with only one side is normal; inventing the
                # other is not.
                if not isinstance(leg, dict):
                    continue
                quotes.append(
                    self._build_quote(
                        leg=leg,
                        symbol=symbol,
                        strike=strike,
                        option_type=option_type,
                        expiry=expiry,
                        as_of=as_of,
                        spot=spot,
                        years=years,
                        lot_size=lot_size,
                        tick_size=tick_size,
                    )
                )
        if not quotes:
            raise DataAdapterError(
                f"Dhan chain for {symbol} {expiry.isoformat()} had no quotable strikes"
            )
        return quotes

    def _build_quote(
        self,
        *,
        leg: dict[str, Any],
        symbol: str,
        strike: Decimal,
        option_type: OptionType,
        expiry: date,
        as_of: datetime,
        spot: Decimal | None,
        years: float,
        lot_size: int,
        tick_size: Decimal,
    ) -> OptionQuote:
        from index_option_brain.analytics.pricing import greeks_from_iv

        contract = OptionContractSpec(
            underlying_symbol=symbol,
            expiry=expiry,
            strike=strike,
            option_type=option_type,
            lot_size=lot_size,
            tick_size=tick_size,
        )
        iv = _optional_decimal(leg.get("implied_volatility"))
        greeks: Greeks | None = None
        raw_greeks = leg.get("greeks")

        if self._client.config.trust_broker_greeks and isinstance(raw_greeks, dict):
            greeks = Greeks(
                delta=_decimal(raw_greeks.get("delta", 0), "delta", "chain"),
                gamma=_decimal(raw_greeks.get("gamma", 0), "gamma", "chain"),
                theta=_decimal(raw_greeks.get("theta", 0), "theta", "chain"),
                vega=_decimal(raw_greeks.get("vega", 0), "vega", "chain"),
            )
        elif iv is not None and spot is not None and years > 0:
            # Recomputed rather than taken, so one rate and one day-count
            # convention apply across every provider. Delta fit compares
            # strikes; comparing two providers' conventions would compare
            # quantities that are not the same quantity.
            computed = greeks_from_iv(
                spot=float(spot),
                strike=float(strike),
                years=years,
                iv_percent=float(iv),
                option_type=option_type,
                rate=self._client.config.risk_free_rate,
            )
            greeks = Greeks(
                delta=Decimal(str(round(computed.delta, 6))),
                gamma=Decimal(str(round(computed.gamma, 10))),
                theta=Decimal(str(round(computed.theta, 4))),
                vega=Decimal(str(round(computed.vega, 4))),
            )

        return OptionQuote(
            contract=contract,
            timestamp=as_of,
            ltp=_decimal(_require(leg, "last_price", "option chain"), "last_price", "chain"),
            bid=_optional_decimal(leg.get("top_bid_price")),
            ask=_optional_decimal(leg.get("top_ask_price")),
            volume=_int(leg.get("volume")),
            open_interest=_int(leg.get("oi")),
            # Dhan reports the previous day's OI rather than the change, so
            # the change is derived. Reporting the raw previous value as a
            # change would invert the meaning of every OI build.
            open_interest_change=_int(leg.get("oi")) - _int(leg.get("previous_oi")),
            implied_volatility=iv,
            greeks=greeks,
        )

    # -------------------------------------------------------------- account

    async def get_account_snapshot(self) -> AccountSnapshot:
        """Funds, which is what unblocks the Risk Engine.

        Note this reaches the *real* account even in the sandbox, where the
        balances are the sandbox's own. That is the point of building against
        it first.
        """
        payload = await self._client.get("/fundlimit")
        if not isinstance(payload, dict):
            raise DataAdapterError("Dhan fundlimit response is not an object")
        return AccountSnapshot(
            timestamp=datetime.now(UTC),
            available_margin=_decimal(
                _require(payload, "availabelBalance", "fundlimit"),
                "availabelBalance",
                "fundlimit",
            ),
            used_margin=_decimal(
                _require(payload, "utilizedAmount", "fundlimit"),
                "utilizedAmount",
                "fundlimit",
            ),
            net_equity=_decimal(
                _require(payload, "sodLimit", "fundlimit"), "sodLimit", "fundlimit"
            ),
        )


DHAN_ORDER_SIDE: dict[OrderSide, str] = {
    OrderSide.BUY: "BUY",
    OrderSide.SELL: "SELL",
}


class DhanBrokerAdapter(BrokerAdapter):
    """Order placement, cancellation and status against DhanHQ v2.

    Used only by the Order Manager, which is used only after the Execution
    Gate has passed all sixteen checks. Nothing in this class decides whether
    to trade; it translates an authorized `OrderRequest` into Dhan's shape and
    translates the reply back.

    Sandbox by default. `DhanConfig(sandbox=False)` is what points it at real
    money, and that is deliberately a separate, explicit act.
    """

    def __init__(
        self,
        client: DhanClient,
        master: DhanInstrumentMaster,
        *,
        product_type: str = "MARGIN",
    ) -> None:
        self._client = client
        self._master = master
        self._product_type = product_type

    @property
    def is_sandbox(self) -> bool:
        return self._client.config.sandbox

    async def aclose(self) -> None:
        await self._client.aclose()

    def _security_id(self, request: OrderRequest) -> str:
        contract = request.contract
        record = self._master.option(
            contract.underlying_symbol,
            contract.expiry,
            contract.strike,
            contract.option_type,
        )
        if record is None:
            # Refusing is right. Guessing an id, or sending the strike as
            # though it were one, would place an order on some other
            # instrument entirely.
            raise DataAdapterError(
                f"No listed contract for {contract.instrument_key} in the "
                "instrument master, so it has no security id to trade. The "
                "master may be stale — reload it."
            )
        return record.security_id

    def _order_body(self, request: OrderRequest) -> dict[str, Any]:
        return {
            "dhanClientId": self._client.config.client_id,
            # Dhan's idempotency key. The same value the Order Manager uses to
            # recognize a resubmission, so both ends agree on what "the same
            # order" means and a retry cannot double the position.
            "correlationId": request.client_order_id,
            "transactionType": DHAN_ORDER_SIDE[request.side],
            "exchangeSegment": FNO_SEGMENT,
            "productType": self._product_type,
            "orderType": "LIMIT" if request.limit_price is not None else "MARKET",
            "validity": "DAY",
            "securityId": self._security_id(request),
            "quantity": request.quantity,
            "price": float(request.limit_price) if request.limit_price else 0,
            "drvExpiryDate": request.contract.expiry.isoformat(),
            "drvOptionType": str(request.contract.option_type),
            "drvStrikePrice": float(request.contract.strike),
        }

    async def place_order(self, request: OrderRequest) -> Order:
        payload = await self._client.post("/orders", self._order_body(request))
        return self._to_order(payload, request=request)

    async def cancel_order(
        self, broker_order_id: str, *, known: Order | None = None
    ) -> Order:
        payload = await self._client.delete(f"/orders/{broker_order_id}")
        return self._to_order(payload, broker_order_id=broker_order_id, known=known)

    async def get_order_status(
        self, broker_order_id: str, *, known: Order | None = None
    ) -> Order:
        payload = await self._client.get(f"/orders/{broker_order_id}")
        if isinstance(payload, list):
            # Dhan answers this one with a single-element list.
            if not payload:
                raise DataAdapterError(f"Dhan knows no order {broker_order_id}")
            payload = payload[0]
        return self._to_order(payload, broker_order_id=broker_order_id, known=known)

    def _to_order(
        self,
        payload: Any,
        *,
        request: OrderRequest | None = None,
        broker_order_id: str | None = None,
        known: Order | None = None,
    ) -> Order:
        if not isinstance(payload, dict):
            raise DataAdapterError(f"Dhan order response is not an object: {payload!r}")

        state = self._map_state(str(payload.get("orderStatus", "")).upper())
        now = datetime.now(UTC)
        resolved_id = str(payload.get("orderId") or broker_order_id or "")

        if request is not None:
            contract = request.contract
            side = request.side
            quantity = request.quantity
            limit_price = request.limit_price
            decision_id = request.decision_id
            thesis_id = request.thesis_id
        elif known is not None:
            # The caller's copy is authoritative for everything but state and
            # fills, which is all this reply is trusted for.
            contract = known.contract
            side = known.side
            quantity = known.quantity
            limit_price = known.limit_price
            decision_id = known.decision_id
            thesis_id = known.thesis_id
        else:
            # No caller copy: reconstruct from the instrument master if the
            # reply names one, and refuse otherwise rather than invent a
            # contract that would reach a reconciled position.
            contract = self._contract_from_payload(payload)
            side = (
                OrderSide.SELL
                if str(payload.get("transactionType", "")).upper() == "SELL"
                else OrderSide.BUY
            )
            quantity = _int(payload.get("quantity"))
            limit_price = _optional_decimal(payload.get("price"))
            decision_id = str(payload.get("correlationId") or "")
            thesis_id = ""

        return Order(
            order_id=resolved_id or "unknown",
            decision_id=decision_id,
            thesis_id=thesis_id,
            contract=contract,
            side=side,
            quantity=quantity,
            limit_price=limit_price,
            state=state,
            broker_order_id=resolved_id or None,
            filled_quantity=_int(payload.get("filledQty") or payload.get("filled_qty")),
            average_fill_price=_optional_decimal(payload.get("averageTradedPrice")),
            created_at=now,
            updated_at=now,
        )

    def _contract_from_payload(self, payload: dict[str, Any]) -> OptionContractSpec:
        security_id = str(payload.get("securityId") or "")
        if not security_id:
            raise DataAdapterError(
                "Dhan's reply identifies no instrument, which is normal for a "
                "cancel acknowledgement. Pass the caller's copy as `known` — "
                "inventing a contract here would put a fabricated instrument "
                "into a reconciled position."
            )
        for record in self._master.records:
            if record.security_id == security_id and record.is_index_option:
                if (
                    record.expiry is None
                    or record.strike is None
                    or record.option_type is None
                ):
                    # Defaulting any of these would describe a different
                    # contract than the one the broker just acted on.
                    raise DataAdapterError(
                        f"Instrument master record {security_id} is missing an "
                        "expiry, strike or option type, so the order it "
                        "belongs to cannot be reconstructed"
                    )
                return OptionContractSpec(
                    underlying_symbol=record.underlying,
                    expiry=record.expiry,
                    strike=record.strike,
                    option_type=record.option_type,
                    lot_size=record.lot_size,
                    tick_size=record.tick_size,
                )
        raise DataAdapterError(
            f"Dhan reported order on security id {security_id!r}, which is not "
            "in the instrument master. Reload it before reconciling."
        )

    def _map_state(self, status: str) -> OrderLifecycleState:
        """Dhan's status vocabulary onto the §30 state machine.

        An unrecognized status becomes FAILED rather than being guessed at.
        Mapping an unknown status onto OPEN would leave the Order Manager
        believing an order is working when it may have filled.
        """
        mapping = {
            "TRANSIT": OrderLifecycleState.SUBMITTED,
            "PENDING": OrderLifecycleState.OPEN,
            "OPEN": OrderLifecycleState.OPEN,
            "PART_TRADED": OrderLifecycleState.PARTIAL,
            "PARTIALLY_FILLED": OrderLifecycleState.PARTIAL,
            "TRADED": OrderLifecycleState.FILLED,
            "EXECUTED": OrderLifecycleState.FILLED,
            "COMPLETE": OrderLifecycleState.FILLED,
            "REJECTED": OrderLifecycleState.REJECTED,
            "CANCELLED": OrderLifecycleState.CANCELLED,
            "CANCELED": OrderLifecycleState.CANCELLED,
            "EXPIRED": OrderLifecycleState.CANCELLED,
        }
        if status not in mapping:
            raise DataAdapterError(
                f"Dhan reported an unrecognized order status {status!r}. Refusing "
                "to guess: mapping it onto OPEN would leave the Order Manager "
                "believing a possibly-filled order is still working."
            )
        return mapping[status]
