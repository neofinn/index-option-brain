"""Delta Exchange adapter, against payloads recorded from the live API.

Every fixture here was captured from api.india.delta.exchange, not written
by hand — the public endpoints need no key, so the field names, types and
units are the exchange's own rather than an inference from documentation.
That is the verification the Dhan adapter is still waiting on, and these
tests are what stops it drifting.
"""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from index_option_brain.contracts.enums import BarInterval, OptionType
from index_option_brain.data.adapters.base import DataAdapterError
from index_option_brain.data.adapters.delta_exchange import (
    DeltaConfig,
    DeltaExchangeAdapter,
    expiry_query,
    parse_option_symbol,
)

PAYLOADS = Path(__file__).parent / "payloads"
CHAIN: list[dict[str, Any]] = json.loads((PAYLOADS / "delta_option_chain.json").read_text())
PRODUCTS: list[dict[str, Any]] = json.loads((PAYLOADS / "delta_products.json").read_text())
CANDLES: list[dict[str, Any]] = json.loads((PAYLOADS / "delta_candles.json").read_text())


class FakeClient:
    def __init__(
        self,
        *,
        chain: list[dict[str, Any]] | None = None,
        products: list[dict[str, Any]] | None = None,
        candles: list[dict[str, Any]] | None = None,
    ) -> None:
        self._chain = CHAIN if chain is None else chain
        self._products = PRODUCTS if products is None else products
        self._candles = CANDLES if candles is None else candles
        self.chain_calls: list[str] = []

    def get_products(self, query: Any = None, auth: bool = False) -> Any:
        return self._products

    def option_chain(
        self, underlying_asset_symbol: str, expiry_date: str, auth: bool = False
    ) -> Any:
        self.chain_calls.append(expiry_date)
        return self._chain

    def get_candles(
        self, symbol: str, resolution: str, start: int, end: int, auth: bool = False
    ) -> Any:
        return self._candles


@pytest.fixture
def adapter() -> DeltaExchangeAdapter:
    return DeltaExchangeAdapter(FakeClient(), config=DeltaConfig())


class TestSymbolParsing:
    def test_a_real_symbol_decomposes(self) -> None:
        kind, asset, strike, expiry = parse_option_symbol("C-BTC-79800-060926")
        assert kind is OptionType.CE
        assert asset == "BTC"
        assert strike == Decimal(79800)
        assert expiry == date(2026, 9, 6)

    def test_puts_are_recognised(self) -> None:
        kind, _, _, _ = parse_option_symbol("P-BTC-80000-060926")
        assert kind is OptionType.PE

    def test_an_unrecognised_symbol_is_refused_not_guessed(self) -> None:
        """Guessing at an unrecognised instrument is how the wrong contract
        gets traded."""
        for bad in ("BTC-79800", "X-BTC-79800-060926", "C-BTC-79800-6926", ""):
            with pytest.raises(DataAdapterError, match="does not match"):
                parse_option_symbol(bad)

    def test_the_expiry_query_uses_the_venues_format(self) -> None:
        assert expiry_query(date(2026, 9, 6)) == "06-09-2026"


class TestUnits:
    """Prices and greeks are per one unit of the underlying; one contract is
    `contract_value` of it. Multiplying in the wrong place is the same class
    of error as reading NIFTY's lot as 75 when it is 65, and it is silent."""

    async def test_the_multiplier_travels_with_the_quote(
        self, adapter: DeltaExchangeAdapter
    ) -> None:
        chain = await adapter.get_option_chain("BTC", date(2026, 9, 6))
        assert chain
        for quote in chain:
            assert quote.contract_multiplier == Decimal("0.001")

    async def test_a_premium_is_per_unit_of_underlying(
        self, adapter: DeltaExchangeAdapter
    ) -> None:
        """A mid of ~1,600 on a BTC call is USD per BTC, so one contract is
        about 1.60. Anyone reading the mid as the contract price is out by
        1,000x."""
        chain = await adapter.get_option_chain("BTC", date(2026, 9, 6))
        call = next(q for q in chain if q.contract.option_type is OptionType.CE)
        per_contract = call.mid * call.contract_multiplier
        assert call.mid > 100
        assert per_contract < 10

    async def test_the_tick_size_is_read_not_assumed(
        self, adapter: DeltaExchangeAdapter
    ) -> None:
        spec = await adapter.get_index_spec("BTC")
        assert spec.tick_size == Decimal("0.1")

    async def test_the_strike_step_is_inferred_from_the_listing(
        self, adapter: DeltaExchangeAdapter
    ) -> None:
        spec = await adapter.get_index_spec("BTC")
        assert spec.strike_step == Decimal(200)

    async def test_an_irregular_listing_yields_no_strike_step(self) -> None:
        """A single assumed step would place strikes that do not exist."""
        sparse = [
            {"state": "live", "strike_price": "1000", "tick_size": "0.1",
             "underlying_asset": {"symbol": "BTC"}},
            {"state": "live", "strike_price": "2000", "tick_size": "0.1",
             "underlying_asset": {"symbol": "BTC"}},
        ]
        adapter = DeltaExchangeAdapter(FakeClient(products=sparse))
        assert (await adapter.get_index_spec("BTC")).strike_step is None


class TestImpliedVolatility:
    async def test_iv_is_converted_to_percentage_points(
        self, adapter: DeltaExchangeAdapter
    ) -> None:
        """Delta quotes 0.31925; this system carries 31.925 everywhere."""
        chain = await adapter.get_option_chain("BTC", date(2026, 9, 6))
        ivs = [float(q.implied_volatility) for q in chain if q.implied_volatility]
        assert ivs
        assert all(5 < iv < 300 for iv in ivs)

    async def test_a_mark_iv_outside_the_bid_ask_range_is_refused(self) -> None:
        """Having three IVs is what makes the published one checkable — the
        same stale-quote problem NSE's single IV had, except detectable."""
        row = dict(CHAIN[0])
        row["quotes"] = {
            **(row.get("quotes") or {}),
            "bid_iv": "0.30",
            "ask_iv": "0.32",
            "mark_iv": "0.90",
        }
        adapter = DeltaExchangeAdapter(FakeClient(chain=[row]))
        chain = await adapter.get_option_chain("BTC", date(2026, 9, 6))
        assert chain[0].implied_volatility is None

    async def test_a_mark_iv_inside_the_range_is_accepted(self) -> None:
        row = dict(CHAIN[0])
        row["quotes"] = {
            **(row.get("quotes") or {}),
            "bid_iv": "0.30",
            "ask_iv": "0.34",
            "mark_iv": "0.32",
        }
        adapter = DeltaExchangeAdapter(FakeClient(chain=[row]))
        chain = await adapter.get_option_chain("BTC", date(2026, 9, 6))
        assert chain[0].implied_volatility == Decimal(32)


class TestGreeks:
    async def test_published_greeks_reach_the_quote(
        self, adapter: DeltaExchangeAdapter
    ) -> None:
        """The venue's main advantage over NSE, which publishes none."""
        chain = await adapter.get_option_chain("BTC", date(2026, 9, 6))
        assert all(q.greeks is not None for q in chain)

    async def test_call_and_put_deltas_have_the_right_signs(
        self, adapter: DeltaExchangeAdapter
    ) -> None:
        chain = await adapter.get_option_chain("BTC", date(2026, 9, 6))
        for quote in chain:
            assert quote.greeks is not None
            if quote.contract.option_type is OptionType.CE:
                assert float(quote.greeks.delta) > 0
            else:
                assert float(quote.greeks.delta) < 0

    async def test_a_disagreeing_published_delta_is_replaced(self) -> None:
        """Two independent derivations disagreeing is a real check, unlike
        the lot-size case where both sides came from one constant and the
        comparison could never fail."""
        row = dict(CHAIN[0])
        row["greeks"] = {**(row.get("greeks") or {}), "delta": "0.99"}
        adapter = DeltaExchangeAdapter(FakeClient(chain=[row]))
        chain = await adapter.get_option_chain("BTC", date(2026, 9, 6))

        assert chain[0].greeks is not None
        assert float(chain[0].greeks.delta) != pytest.approx(0.99)

    async def test_the_cross_check_can_be_turned_off(self) -> None:
        row = dict(CHAIN[0])
        row["greeks"] = {**(row.get("greeks") or {}), "delta": "0.99"}
        adapter = DeltaExchangeAdapter(
            FakeClient(chain=[row]), config=DeltaConfig(compute_greeks=False)
        )
        chain = await adapter.get_option_chain("BTC", date(2026, 9, 6))
        assert chain[0].greeks is not None
        assert float(chain[0].greeks.delta) == pytest.approx(0.99)


class TestCandles:
    async def test_real_history_comes_back_oldest_first(
        self, adapter: DeltaExchangeAdapter
    ) -> None:
        """The thing NSE will not serve at all."""
        bars = await adapter.get_index_bars("BTC", BarInterval.DAY, 30)
        assert len(bars) == 30
        assert bars == sorted(bars, key=lambda b: b.timestamp)

    async def test_fewer_bars_than_asked_for_is_not_padded(
        self, adapter: DeltaExchangeAdapter
    ) -> None:
        bars = await adapter.get_index_bars("BTC", BarInterval.DAY, 500)
        assert 0 < len(bars) <= len(CANDLES)

    async def test_an_unsupported_interval_is_refused(
        self, adapter: DeltaExchangeAdapter
    ) -> None:
        with pytest.raises(DataAdapterError, match="no .* candles"):
            await adapter.get_index_bars("BTC", "4h", 10)  # type: ignore[arg-type]

    async def test_zero_bars_requested_is_empty_not_everything(
        self, adapter: DeltaExchangeAdapter
    ) -> None:
        assert await adapter.get_index_bars("BTC", BarInterval.DAY, 0) == []


class TestQuoteAndExpiries:
    async def test_spot_comes_from_the_chain_the_greeks_were_computed_against(
        self, adapter: DeltaExchangeAdapter
    ) -> None:
        """Reading spot from a separate ticker call would let the quote and
        the greeks describe different moments."""
        quote = await adapter.get_index_quote("BTC")
        assert quote.ltp == Decimal(str(CHAIN[0]["spot_price"]))

    async def test_expiries_are_sorted_and_live_only(
        self, adapter: DeltaExchangeAdapter
    ) -> None:
        expiries = await adapter.get_available_expiries("BTC")
        assert expiries == sorted(expiries)
        assert all(isinstance(e, date) for e in expiries)

    async def test_an_empty_chain_is_refused(self) -> None:
        adapter = DeltaExchangeAdapter(FakeClient(chain=[]))
        with pytest.raises(DataAdapterError, match="empty chain"):
            await adapter.get_option_chain("BTC", date(2026, 9, 6))

    async def test_no_live_products_is_refused(self) -> None:
        adapter = DeltaExchangeAdapter(FakeClient(products=[]))
        with pytest.raises(DataAdapterError, match="no live option products"):
            await adapter.get_index_spec("BTC")

    async def test_a_client_failure_becomes_a_data_adapter_error(self) -> None:
        class Broken(FakeClient):
            def option_chain(self, **kw: Any) -> Any:
                raise RuntimeError("connection reset")

        adapter = DeltaExchangeAdapter(Broken())
        with pytest.raises(DataAdapterError, match="Delta request failed"):
            await adapter.get_option_chain("BTC", date(2026, 9, 6))


class TestDefaults:
    def test_the_default_environment_is_testnet(self) -> None:
        """The safe environment is the one you get without asking."""
        assert "testnet" in DeltaConfig().base_url.lower()
