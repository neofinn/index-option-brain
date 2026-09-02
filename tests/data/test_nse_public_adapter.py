"""Tests for the live NSE public adapter, against recorded exchange payloads.

The transport is a test double; the payloads are not (see
`recorded/README.md`). So a failure here means the adapter misreads NSE, not
that a fixture drifted from a fiction.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from index_option_brain.contracts.enums import BarInterval, OptionType
from index_option_brain.contracts.provider import Capability, ProviderKind
from index_option_brain.data.adapters.base import (
    DataAdapterError,
    IndexDataAdapter,
    OptionsChainAdapter,
    VolatilityDataAdapter,
)
from index_option_brain.data.adapters.nse_public import (
    IST,
    NSE_PUBLIC_DESCRIPTOR,
    NseIndexConfig,
    NsePublicAdapter,
    expiry_instant,
    infer_strike_step,
    parse_ist_timestamp,
    parse_nse_date,
    years_to_expiry,
)
from index_option_brain.data.http import HttpError, HttpResponse, RecordedSession
from tests.data.conftest import nse_session, payload

NEAR_EXPIRY = date(2026, 9, 8)
SPOT = Decimal("23914.45")


def ce(quotes, strike: int):
    return next(
        q
        for q in quotes
        if q.contract.strike == strike and q.contract.option_type is OptionType.CE
    )


def pe(quotes, strike: int):
    return next(
        q
        for q in quotes
        if q.contract.strike == strike and q.contract.option_type is OptionType.PE
    )


class TestItIsAnAdapter:
    def test_it_implements_the_capability_interfaces_it_claims(self):
        assert issubclass(NsePublicAdapter, IndexDataAdapter)
        assert issubclass(NsePublicAdapter, OptionsChainAdapter)
        assert issubclass(NsePublicAdapter, VolatilityDataAdapter)

    def test_it_is_not_an_account_or_broker_adapter(self):
        """A data source must not be selectable as an execution route."""
        assert NSE_PUBLIC_DESCRIPTOR.kind is ProviderKind.DATA
        assert not NSE_PUBLIC_DESCRIPTOR.can_trade
        assert not hasattr(NsePublicAdapter, "get_account_snapshot")
        assert not hasattr(NsePublicAdapter, "place_order")

    def test_the_descriptor_declares_only_what_nse_serves(self):
        assert NSE_PUBLIC_DESCRIPTOR.implemented
        assert NSE_PUBLIC_DESCRIPTOR.supports(
            Capability.INDEX_QUOTE,
            Capability.OPTION_CHAIN,
            Capability.EXPIRY_LIST,
            Capability.INDIA_VIX,
        )

    @pytest.mark.parametrize(
        "absent",
        [
            Capability.INDEX_BARS,
            Capability.CONSTITUENT_QUOTES,
            Capability.OPTION_GREEKS,
            Capability.ACCOUNT_SNAPSHOT,
            Capability.ORDER_PLACEMENT,
        ],
    )
    def test_the_descriptor_does_not_claim_what_nse_withholds(self, absent):
        """Each of these was probed against the live endpoint and does not
        work. Declaring one would let the console offer a control that cannot
        function."""
        assert absent not in NSE_PUBLIC_DESCRIPTOR.capabilities


class TestIndexQuote:
    async def test_it_reads_the_recorded_nifty_snapshot(self, nse: NsePublicAdapter):
        quote = await nse.get_index_quote("NIFTY")
        assert quote.symbol == "NIFTY"
        assert quote.ltp == Decimal("23914.45")
        assert quote.open == Decimal(23858)
        assert quote.high == Decimal("23914.45")
        assert quote.low == Decimal("23786.8")
        assert quote.previous_close == Decimal("24055.8")

    async def test_change_pct_is_computed_against_the_previous_close(
        self, nse: NsePublicAdapter
    ):
        quote = await nse.get_index_quote("NIFTY")
        assert float(quote.change_pct) == pytest.approx(-0.5876, abs=1e-4)

    async def test_the_timestamp_is_converted_from_ist_to_utc(
        self, nse: NsePublicAdapter
    ):
        """NSE sends "02-Sep-2026 15:30" with no zone. 15:30 IST is 10:00 UTC,
        and everything downstream computes time-to-expiry from this."""
        quote = await nse.get_index_quote("NIFTY")
        assert quote.timestamp == datetime(2026, 9, 2, 10, 0, tzinfo=UTC)
        assert quote.timestamp.tzinfo is not None

    async def test_vwap_is_none_rather_than_invented(self, nse: NsePublicAdapter):
        """NSE publishes no index VWAP here. None means "not measured"; a
        stand-in would corrupt the VWAP relationship the Index brain reads."""
        quote = await nse.get_index_quote("NIFTY")
        assert quote.vwap is None

    async def test_banknifty_maps_to_its_nse_index_name(self, nse: NsePublicAdapter):
        """The adapter has to translate: NSE calls it "NIFTY BANK"."""
        quote = await nse.get_index_quote("BANKNIFTY")
        assert quote.symbol == "BANKNIFTY"
        assert quote.ltp == Decimal(57172)
        assert quote.previous_close == Decimal("57409.6")

    async def test_an_unconfigured_symbol_raises_with_the_remedy(
        self, nse: NsePublicAdapter
    ):
        with pytest.raises(DataAdapterError) as exc:
            await nse.get_index_quote("FINNIFTY")
        assert "not configured" in str(exc.value)
        assert "index_config" in str(exc.value)

    async def test_a_symbol_missing_from_the_payload_raises(self):
        """Configured but absent from the response is a broken feed, not an
        empty market."""
        adapter = NsePublicAdapter(
            nse_session(),
            index_config={
                "NIFTY": NseIndexConfig(
                    nse_index_name="NIFTY NEXT 50",
                    display_name="Nifty Next 50",
                    lot_size=25,
                    strike_step=Decimal(50),
                )
            },
        )
        with pytest.raises(DataAdapterError, match="does not contain"):
            await adapter.get_index_quote("NIFTY")


class TestIndexSpec:
    async def test_the_spec_carries_the_configured_contract_size(
        self, nse: NsePublicAdapter
    ):
        spec = await nse.get_index_spec("NIFTY")
        assert spec.symbol == "NIFTY"
        assert spec.name == "Nifty 50"
        assert spec.lot_size == 75
        assert spec.strike_step == Decimal(50)
        assert spec.tick_size == Decimal("0.05")

    async def test_banknifty_has_its_own_strike_step(self, nse: NsePublicAdapter):
        spec = await nse.get_index_spec("BANKNIFTY")
        assert spec.lot_size == 30
        assert spec.strike_step == Decimal(100)

    async def test_the_lot_size_is_overridable(self):
        """Lot sizes are revised by exchange circular and NSE's public
        endpoints do not publish them, so overriding must not require editing
        the package."""
        adapter = NsePublicAdapter(
            nse_session(),
            index_config={
                "NIFTY": NseIndexConfig(
                    nse_index_name="NIFTY 50",
                    display_name="Nifty 50",
                    lot_size=50,
                    strike_step=Decimal(50),
                )
            },
        )
        assert (await adapter.get_index_spec("NIFTY")).lot_size == 50

    async def test_reading_the_spec_makes_no_request(self, session: RecordedSession):
        """The spec is configuration, so it must not depend on the network
        being up."""
        adapter = NsePublicAdapter(session)
        await adapter.get_index_spec("NIFTY")
        assert session.requests == []


class TestIndiaVix:
    async def test_it_reads_india_vix_and_its_previous_close(
        self, nse: NsePublicAdapter
    ):
        current, previous = await nse.get_india_vix()
        assert current == pytest.approx(11.34)
        assert previous == pytest.approx(11.49)

    async def test_vix_and_spot_come_from_one_request(self, session: RecordedSession):
        """Both live on `/api/allIndices`, and a cached snapshot is what keeps
        them describing the same instant. Two separate reads landing on two
        snapshots would build a MarketState that never existed."""
        adapter = NsePublicAdapter(session)
        await adapter.get_index_quote("NIFTY")
        await adapter.get_india_vix()
        api_calls = [r for r in session.requests if "allIndices" in r]
        assert len(api_calls) == 1

    async def test_the_cache_can_be_disabled(self, session: RecordedSession):
        adapter = NsePublicAdapter(session, snapshot_ttl_seconds=0.0)
        await adapter.get_index_quote("NIFTY")
        await adapter.get_india_vix()
        assert len([r for r in session.requests if "allIndices" in r]) == 2


class TestExpiries:
    async def test_it_reads_the_expiry_list(self, nse: NsePublicAdapter):
        expiries = await nse.get_available_expiries("NIFTY")
        assert expiries[:3] == [
            date(2026, 9, 8),
            date(2026, 9, 15),
            date(2026, 9, 22),
        ]

    async def test_weekly_expiries_are_tuesdays(self, nse: NsePublicAdapter):
        """NSE moved weekly index expiry off Thursday. Hardcoding Thursday
        anywhere would silently mis-price every weekly by two days."""
        expiries = await nse.get_available_expiries("NIFTY")
        assert [e.strftime("%A") for e in expiries[:4]] == ["Tuesday"] * 4

    async def test_expiries_come_back_sorted(self, nse: NsePublicAdapter):
        """The brains treat the first entry as the near expiry, and the
        endpoint's ordering is not a documented guarantee."""
        expiries = await nse.get_available_expiries("NIFTY")
        assert expiries == sorted(expiries)

    async def test_an_empty_expiry_list_raises(self):
        adapter = NsePublicAdapter(
            nse_session(**{"contract-info": json.dumps({"expiryDates": []})})
        )
        with pytest.raises(DataAdapterError, match="no expiry dates"):
            await adapter.get_available_expiries("NIFTY")


class TestOptionChain:
    async def test_it_reads_both_sides_of_every_recorded_strike(
        self, nse: NsePublicAdapter
    ):
        chain = await nse.get_option_chain("NIFTY", NEAR_EXPIRY)
        assert len(chain) == 16
        assert len({q.contract.strike for q in chain}) == 8

    async def test_contract_specs_are_normalized(self, nse: NsePublicAdapter):
        chain = await nse.get_option_chain("NIFTY", NEAR_EXPIRY)
        leg = ce(chain, 23900)
        assert leg.contract.underlying_symbol == "NIFTY"
        assert leg.contract.expiry == NEAR_EXPIRY
        assert leg.contract.strike == Decimal(23900)
        assert leg.contract.option_type is OptionType.CE
        assert leg.contract.lot_size == 75

    async def test_bid_and_ask_come_from_the_depth_fields(
        self, nse: NsePublicAdapter
    ):
        """NSE leaves `bidprice`/`askPrice` null and puts the real top of book
        in `buyPrice1`/`sellPrice1`. Reading the documented-looking fields
        would produce a chain with no bid or ask anywhere."""
        chain = await nse.get_option_chain("NIFTY", NEAR_EXPIRY)
        leg = ce(chain, 23900)
        assert leg.bid == Decimal("130.15")
        assert leg.ask == Decimal("131.6")
        assert leg.mid == Decimal("130.875")

    async def test_ltp_open_interest_and_volume_are_read(self, nse: NsePublicAdapter):
        chain = await nse.get_option_chain("NIFTY", NEAR_EXPIRY)
        leg = ce(chain, 23900)
        assert leg.ltp == Decimal("131.6")
        assert leg.open_interest == 95996
        assert leg.open_interest_change == 74969
        assert leg.volume == 2635119

    async def test_open_interest_change_keeps_its_sign(self):
        """Unwinding shows as a negative change in OI, and the Options brain
        reads the direction rather than the magnitude — an absolute value here
        would turn every unwind into a build. The recorded snapshot happens to
        contain no negative values, so this uses a hand-built payload to
        exercise the sign path."""
        adapter = NsePublicAdapter(
            nse_session(
                **{
                    "option-chain-v3": json.dumps(
                        {
                            "records": {
                                "timestamp": "02-Sep-2026 15:40:00",
                                "underlyingValue": 23914.45,
                                "data": [
                                    {
                                        "strikePrice": 23900,
                                        "CE": {
                                            "lastPrice": 131.6,
                                            "buyPrice1": 130.15,
                                            "sellPrice1": 131.6,
                                            "openInterest": 95996,
                                            "changeinOpenInterest": -41250,
                                            "totalTradedVolume": 2635119,
                                            "impliedVolatility": 8.39,
                                        },
                                    }
                                ],
                            }
                        }
                    )
                }
            )
        )
        chain = await adapter.get_option_chain("NIFTY", NEAR_EXPIRY)
        assert chain[0].open_interest_change == -41250

    async def test_recorded_open_interest_values_round_trip_exactly(
        self, nse: NsePublicAdapter
    ):
        chain = await nse.get_option_chain("NIFTY", NEAR_EXPIRY)
        assert {
            (int(q.contract.strike), q.contract.option_type.value): q.open_interest_change
            for q in chain
            if q.contract.strike in (Decimal(23900), Decimal(23850))
        } == {
            (23900, "CE"): 74969,
            (23900, "PE"): 39025,
            (23850, "CE"): 31296,
            (23850, "PE"): 47698,
        }

    async def test_the_timestamp_is_the_payloads_own(self, nse: NsePublicAdapter):
        """Every quote is stamped with the chain's timestamp, not wall clock:
        a delayed or replayed snapshot must not acquire a fresh time."""
        chain = await nse.get_option_chain("NIFTY", NEAR_EXPIRY)
        expected = datetime(2026, 9, 2, 10, 10, tzinfo=UTC)  # 15:40 IST
        assert {q.timestamp for q in chain} == {expected}

    async def test_relative_spread_is_computed(self, nse: NsePublicAdapter):
        chain = await nse.get_option_chain("NIFTY", NEAR_EXPIRY)
        spread = ce(chain, 23900).relative_spread
        assert spread is not None
        assert float(spread) == pytest.approx(0.01108, abs=1e-4)

    async def test_an_empty_chain_raises(self):
        adapter = NsePublicAdapter(
            nse_session(
                **{
                    "option-chain-v3": json.dumps(
                        {"records": {"data": [], "timestamp": "02-Sep-2026 15:40:00"}}
                    )
                }
            )
        )
        with pytest.raises(DataAdapterError, match="empty chain"):
            await adapter.get_option_chain("NIFTY", NEAR_EXPIRY)

    async def test_a_chain_without_a_timestamp_raises(self):
        adapter = NsePublicAdapter(
            nse_session(
                **{
                    "option-chain-v3": json.dumps(
                        {
                            "records": {
                                "data": [{"strikePrice": 23900}],
                                "underlyingValue": 23914.45,
                            }
                        }
                    )
                }
            )
        )
        with pytest.raises(DataAdapterError, match="no timestamp"):
            await adapter.get_option_chain("NIFTY", NEAR_EXPIRY)

    async def test_a_strike_quoted_on_one_side_only_is_not_invented(self):
        """A hand-built payload, not recorded data: it exercises the parser
        branch for a strike NSE lists with only one side. The correct result
        is one quote, not two."""
        adapter = NsePublicAdapter(
            nse_session(
                **{
                    "option-chain-v3": json.dumps(
                        {
                            "records": {
                                "timestamp": "02-Sep-2026 15:40:00",
                                "underlyingValue": 23914.45,
                                "data": [
                                    {
                                        "strikePrice": 23900,
                                        "CE": {
                                            "lastPrice": 131.6,
                                            "buyPrice1": 130.15,
                                            "sellPrice1": 131.6,
                                            "openInterest": 95996,
                                            "changeinOpenInterest": 74969,
                                            "totalTradedVolume": 2635119,
                                            "impliedVolatility": 8.39,
                                        },
                                    }
                                ],
                            }
                        }
                    )
                }
            )
        )
        chain = await adapter.get_option_chain("NIFTY", NEAR_EXPIRY)
        assert len(chain) == 1
        assert chain[0].contract.option_type is OptionType.CE


class TestGreeks:
    async def test_greeks_are_computed_because_nse_publishes_none(
        self, nse: NsePublicAdapter
    ):
        """No Indian source found publishes greeks. Confirm the raw payload
        really has none, so this test fails loudly if NSE ever adds them and
        the adapter should start reading rather than computing."""
        raw = json.loads(payload("nse_option_chain.json"))
        leg = raw["records"]["data"][0]["CE"]
        assert not {"delta", "gamma", "theta", "vega"} & set(leg)

        chain = await nse.get_option_chain("NIFTY", NEAR_EXPIRY)
        assert ce(chain, 23900).greeks is not None

    async def test_at_the_money_delta_is_near_half(self, nse: NsePublicAdapter):
        chain = await nse.get_option_chain("NIFTY", NEAR_EXPIRY)
        assert float(ce(chain, 23900).greeks.delta) == pytest.approx(0.56, abs=0.06)
        assert float(pe(chain, 23900).greeks.delta) == pytest.approx(-0.45, abs=0.06)

    async def test_delta_decreases_across_the_near_atm_strikes(
        self, nse: NsePublicAdapter
    ):
        """Only asserted near the money. Across the whole board the smile
        makes delta non-monotonic in strike, and that is a property of real
        chains rather than a defect."""
        chain = await nse.get_option_chain("NIFTY", NEAR_EXPIRY)
        deltas = [float(ce(chain, k).greeks.delta) for k in (23850, 23900, 23950)]
        assert deltas == sorted(deltas, reverse=True)

    async def test_theta_is_negative_for_long_at_the_money_options(
        self, nse: NsePublicAdapter
    ):
        chain = await nse.get_option_chain("NIFTY", NEAR_EXPIRY)
        assert float(ce(chain, 23900).greeks.theta) < 0
        assert float(pe(chain, 23900).greeks.theta) < 0

    async def test_gamma_and_vega_are_positive(self, nse: NsePublicAdapter):
        chain = await nse.get_option_chain("NIFTY", NEAR_EXPIRY)
        leg = ce(chain, 23900)
        assert float(leg.greeks.gamma) > 0
        assert float(leg.greeks.vega) > 0

    async def test_greeks_can_be_switched_off(self, session: RecordedSession):
        adapter = NsePublicAdapter(session, compute_greeks=False)
        chain = await adapter.get_option_chain("NIFTY", NEAR_EXPIRY)
        assert all(q.greeks is None for q in chain)
        # IV is still reported: not computing greeks is not the same as
        # discarding the volatility reading.
        assert ce(chain, 23900).implied_volatility is not None

    async def test_a_strike_without_iv_has_no_greeks(self, nse: NsePublicAdapter):
        """Greeks without an IV would be greeks off a made-up volatility."""
        chain = await nse.get_option_chain("NIFTY", NEAR_EXPIRY)
        for quote in chain:
            if quote.implied_volatility is None:
                assert quote.greeks is None


class TestImpliedVolatilityPolicy:
    async def test_iv_is_marked_to_the_live_book_not_the_last_trade(
        self, nse: NsePublicAdapter
    ):
        """NSE publishes 8.39% for the 23900 CE, computed from its LTP of
        131.60. The book stood at 130.15/131.60, so the mid is 130.875 and the
        marked IV is slightly different. Deliberate: a desk marks to mid."""
        raw = json.loads(payload("nse_option_chain.json"))
        published = next(
            row["CE"]["impliedVolatility"]
            for row in raw["records"]["data"]
            if row["strikePrice"] == 23900
        )
        assert published == 8.39

        chain = await nse.get_option_chain("NIFTY", NEAR_EXPIRY)
        marked = ce(chain, 23900).implied_volatility
        assert marked is not None
        assert marked != Decimal("8.39")
        assert Decimal(8) < marked < Decimal(10)

    async def test_a_stale_published_iv_does_not_reach_the_greeks(
        self, nse: NsePublicAdapter
    ):
        """The 22900 CE: NSE published 46.55% off a last trade of 1,190 while
        the book stood at 965.20/1,082.25. 46.55% would give the strike a
        delta of 0.78 and an enormous vega, and if it survived into strike
        ranking it would compete with genuine candidates."""
        raw = json.loads(payload("nse_option_chain.json"))
        published = next(
            row["CE"]["impliedVolatility"]
            for row in raw["records"]["data"]
            if row["strikePrice"] == 22900
        )
        assert published == 46.55

        chain = await nse.get_option_chain("NIFTY", NEAR_EXPIRY)
        assert ce(chain, 22900).implied_volatility is None
        assert ce(chain, 22900).greeks is None

    async def test_iv_is_recovered_when_nse_publishes_none(
        self, nse: NsePublicAdapter
    ):
        """The 23600 CE has a two-sided book 1.20 wide on a 344 mid and no
        published IV at all. Dropping it would throw away a perfectly
        tradeable strike, so the premium is inverted instead."""
        raw = json.loads(payload("nse_option_chain.json"))
        published = next(
            row["CE"]["impliedVolatility"]
            for row in raw["records"]["data"]
            if row["strikePrice"] == 23600
        )
        assert published == 0

        chain = await nse.get_option_chain("NIFTY", NEAR_EXPIRY)
        recovered = ce(chain, 23600).implied_volatility
        assert recovered is not None
        assert Decimal(4) < recovered < Decimal(12)
        assert ce(chain, 23600).greeks is not None

    async def test_an_unmarkable_book_yields_no_iv(self, nse: NsePublicAdapter):
        """The 22300 CE quoted 1,469.75/1,787.80 — 318 points wide, with the
        mid below the European lower bound. No volatility explains it, and
        none is reported."""
        chain = await nse.get_option_chain("NIFTY", NEAR_EXPIRY)
        assert ce(chain, 22300).implied_volatility is None
        assert ce(chain, 22300).greeks is None
        # The quote itself still exists — the price is real, only the
        # volatility reading is unavailable.
        assert ce(chain, 22300).bid == Decimal("1469.75")

    async def test_published_iv_can_be_preferred(self, session: RecordedSession):
        """Escape hatch: pass NSE's own numbers through unchanged, for
        reconciling the console against nseindia.com."""
        adapter = NsePublicAdapter(session, prefer_published_iv=True)
        chain = await adapter.get_option_chain("NIFTY", NEAR_EXPIRY)
        assert ce(chain, 23900).implied_volatility == Decimal("8.39")
        assert ce(chain, 22900).implied_volatility == Decimal("46.55")

    async def test_a_tighter_spread_limit_discards_more_strikes(
        self, session: RecordedSession
    ):
        strict = NsePublicAdapter(session, max_relative_spread_for_iv=0.002)
        chain = await strict.get_option_chain("NIFTY", NEAR_EXPIRY)
        # 23600 marked 344.25/345.45 — a relative spread of ~0.0035, now over
        # the limit, and it carries no published IV to fall back on.
        assert ce(chain, 23600).implied_volatility is None

    async def test_far_wings_keep_a_plausible_smile(self, nse: NsePublicAdapter):
        """The 25600 CE trades at 0.85/0.90 — near the tick floor but a real
        two-sided market, so it must still produce a usable IV."""
        chain = await nse.get_option_chain("NIFTY", NEAR_EXPIRY)
        wing = ce(chain, 25600).implied_volatility
        assert wing is not None
        assert Decimal(10) < wing < Decimal(40)


class TestUnavailableCapabilities:
    async def test_historical_bars_raise_rather_than_being_synthesised(
        self, nse: NsePublicAdapter
    ):
        """NSE's history endpoint blocks automated clients. Building candles
        from the current snapshot would corrupt ATR, RSI and every structural
        level derived from them."""
        with pytest.raises(DataAdapterError) as exc:
            await nse.get_index_bars("NIFTY", BarInterval.DAY, 20)
        assert "no historical bars" in str(exc.value)

    async def test_the_bars_error_names_a_remedy(self, nse: NsePublicAdapter):
        with pytest.raises(DataAdapterError) as exc:
            await nse.get_index_bars("NIFTY", BarInterval.MINUTE_5, 100)
        message = str(exc.value)
        assert "broker adapter" in message
        assert "LiveBarAggregator" in message

    def test_it_does_not_implement_the_constituent_interface(self):
        """`/api/equity-stockIndices` returns 404 to this client, so index
        breadth needs a different provider and the class must not advertise
        otherwise."""
        from index_option_brain.data.adapters.base import ConstituentDataAdapter

        assert not issubclass(NsePublicAdapter, ConstituentDataAdapter)


class TestTransportFailures:
    async def test_a_transport_error_becomes_a_data_adapter_error(self):
        """Callers must have exactly one exception type to catch when they
        decide to withhold trading (spec §29)."""
        adapter = NsePublicAdapter(RecordedSession({}))
        with pytest.raises(DataAdapterError, match="Cannot reach NSE"):
            await adapter.get_index_quote("NIFTY")

    async def test_a_server_error_is_reported_with_its_status(self):
        adapter = NsePublicAdapter(
            nse_session(allIndices=HttpResponse(500, "boom"))
        )
        with pytest.raises(DataAdapterError, match="HTTP 500"):
            await adapter.get_index_quote("NIFTY")

    async def test_an_html_interstitial_is_not_treated_as_data(self):
        """A 200 carrying HTML is the anti-bot page. Parsing it as data is how
        a scraper starts quietly reporting nonsense."""
        adapter = NsePublicAdapter(
            nse_session(
                allIndices=HttpResponse(200, "<html>Access Denied</html>")
            )
        )
        with pytest.raises(DataAdapterError, match="non-JSON"):
            await adapter.get_index_quote("NIFTY")

    async def test_a_rejected_request_is_retried_once_after_re_warming(self):
        """An expired cookie is the one recoverable failure this endpoint has
        in normal operation."""
        session = nse_session()
        first = {"count": 0}
        original = session.get

        async def flaky(url, *, params=None, headers=None):
            if "allIndices" in url and first["count"] == 0:
                first["count"] += 1
                return HttpResponse(403, "denied")
            return await original(url, params=params, headers=headers)

        session.get = flaky  # type: ignore[method-assign]
        adapter = NsePublicAdapter(session)
        quote = await adapter.get_index_quote("NIFTY")
        assert quote.ltp == Decimal("23914.45")
        assert first["count"] == 1

    async def test_a_persistent_rejection_is_not_retried_forever(self):
        session = nse_session(allIndices=HttpResponse(403, "denied"))
        adapter = NsePublicAdapter(session)
        with pytest.raises(DataAdapterError, match="HTTP 403"):
            await adapter.get_index_quote("NIFTY")
        assert len([r for r in session.requests if "allIndices" in r]) == 2

    async def test_a_malformed_number_raises_instead_of_defaulting(self):
        adapter = NsePublicAdapter(
            nse_session(
                allIndices=json.dumps(
                        {
                            "timestamp": "02-Sep-2026 15:30",
                            "data": [
                                {
                                    "index": "NIFTY 50",
                                    "last": "not-a-number",
                                    "open": 1,
                                    "high": 1,
                                    "low": 1,
                                    "previousClose": 1,
                                }
                            ],
                        }
                    )
            )
        )
        with pytest.raises(DataAdapterError, match="not a number"):
            await adapter.get_index_quote("NIFTY")

    async def test_a_missing_field_raises_instead_of_defaulting(self):
        """A zero standing in for a missing previous close would make every
        percentage change in the system wrong."""
        adapter = NsePublicAdapter(
            nse_session(
                allIndices=json.dumps(
                        {
                            "timestamp": "02-Sep-2026 15:30",
                            "data": [
                                {
                                    "index": "NIFTY 50",
                                    "last": 23914.45,
                                    "open": 23858,
                                    "high": 23914.45,
                                    "low": 23786.8,
                                }
                            ],
                        }
                    )
            )
        )
        with pytest.raises(DataAdapterError, match="previousClose"):
            await adapter.get_index_quote("NIFTY")

    async def test_the_session_is_warmed_up_before_the_first_api_call(
        self, session: RecordedSession
    ):
        adapter = NsePublicAdapter(session)
        await adapter.get_index_quote("NIFTY")
        assert session.requests[0] == "https://www.nseindia.com/option-chain"

    async def test_warm_up_happens_once(self, session: RecordedSession):
        adapter = NsePublicAdapter(session, snapshot_ttl_seconds=0.0)
        await adapter.get_index_quote("NIFTY")
        await adapter.get_index_quote("NIFTY")
        warmups = [r for r in session.requests if r.endswith("/option-chain")]
        assert len(warmups) == 1

    async def test_an_injected_session_is_not_closed_by_the_adapter(
        self, session: RecordedSession
    ):
        """The caller owns a session it supplied — several adapters may share
        one, and closing someone else's transport is a surprising side effect."""
        closed = {"value": False}

        async def aclose() -> None:
            closed["value"] = True

        session.aclose = aclose  # type: ignore[method-assign]
        adapter = NsePublicAdapter(session)
        await adapter.aclose()
        assert closed["value"] is False


class TestTimeHelpers:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("02-Sep-2026 15:30", datetime(2026, 9, 2, 10, 0, tzinfo=UTC)),
            ("02-Sep-2026 15:40:00", datetime(2026, 9, 2, 10, 10, tzinfo=UTC)),
            ("02-Sep-2026 09:15:00", datetime(2026, 9, 2, 3, 45, tzinfo=UTC)),
        ],
    )
    def test_ist_timestamps_convert_to_utc(self, raw: str, expected: datetime):
        assert parse_ist_timestamp(raw) == expected

    def test_an_unknown_timestamp_format_raises(self):
        """Falling back to `now()` here would make every time-to-expiry in the
        chain quietly wrong."""
        with pytest.raises(DataAdapterError, match="Unrecognized"):
            parse_ist_timestamp("2026/09/02 15:30 IST")

    @pytest.mark.parametrize(
        "raw", ["08-Sep-2026", "08-09-2026", "2026-09-08"]
    )
    def test_both_nse_date_formats_parse(self, raw: str):
        """The same response family uses "08-Sep-2026" in the expiry list and
        "08-09-2026" inside a chain leg."""
        assert parse_nse_date(raw) == date(2026, 9, 8)

    def test_an_unknown_date_format_raises(self):
        with pytest.raises(DataAdapterError, match="Unrecognized"):
            parse_nse_date("Sept 8 2026")

    def test_expiry_settles_at_the_close_not_midnight(self):
        """A same-day weekly priced to midnight would carry eight and a half
        hours of time value it does not have."""
        instant = expiry_instant(date(2026, 9, 8))
        assert instant == datetime(2026, 9, 8, 10, 0, tzinfo=UTC)
        assert instant.astimezone(IST).hour == 15

    def test_years_to_expiry_is_measured_in_calendar_time(self):
        """Six calendar days, not four trading days: premium decays over the
        weekend, and a Tuesday-expiry weekly held from Friday loses three
        days of value."""
        as_of = datetime(2026, 9, 2, 10, 10, tzinfo=UTC)
        years = years_to_expiry(as_of=as_of, expiry=date(2026, 9, 8))
        assert years == pytest.approx((6 - 10 / 1440) / 365, rel=1e-3)

    def test_years_to_expiry_floors_at_zero_after_settlement(self):
        as_of = datetime(2026, 9, 8, 11, 0, tzinfo=UTC)
        assert years_to_expiry(as_of=as_of, expiry=date(2026, 9, 8)) == 0.0

    def test_an_expiry_day_snapshot_has_hours_not_days_left(self):
        as_of = datetime(2026, 9, 8, 4, 0, tzinfo=UTC)  # 09:30 IST
        years = years_to_expiry(as_of=as_of, expiry=date(2026, 9, 8))
        assert 0 < years * 365 * 24 < 7


class TestStrikeStepInference:
    def test_the_modal_gap_is_the_step(self):
        strikes = [Decimal(k) for k in range(23000, 24001, 50)]
        assert infer_strike_step(strikes) == Decimal(50)

    def test_sparse_legacy_strikes_do_not_move_the_step(self):
        """NSE's strike list mixes a dense band around spot with sparse legacy
        strikes far away, so the mean gap and the max gap are both wrong."""
        strikes = [Decimal(1500), Decimal(3000), Decimal(9000)] + [
            Decimal(k) for k in range(23000, 24001, 50)
        ]
        assert infer_strike_step(strikes) == Decimal(50)

    def test_banknifty_style_hundreds_are_detected(self):
        strikes = [Decimal(k) for k in range(56000, 58001, 100)]
        assert infer_strike_step(strikes) == Decimal(100)

    def test_duplicates_are_ignored(self):
        strikes = [Decimal(23000), Decimal(23000), Decimal(23050), Decimal(23100)]
        assert infer_strike_step(strikes) == Decimal(50)

    def test_too_few_strikes_returns_none(self):
        assert infer_strike_step([Decimal(23000), Decimal(23050)]) is None

    def test_the_recorded_chain_strikes_infer_fifty(self):
        raw = json.loads(payload("nse_contract_info.json"))
        strikes = [Decimal(s) for s in raw["strikePrice"]]
        assert infer_strike_step(strikes) == Decimal(50)


class TestHttpSeam:
    async def test_a_recorded_session_reports_an_unrouted_url(self):
        session = RecordedSession({"/known": "{}"})
        with pytest.raises(HttpError, match="no route"):
            await session.get("https://example.test/unknown")

    def test_a_non_json_body_raises_value_error(self):
        with pytest.raises(ValueError):
            HttpResponse(200, "<html/>").json()

    def test_status_classification(self):
        assert HttpResponse(200, "").is_ok
        assert HttpResponse(204, "").is_ok
        assert not HttpResponse(304, "").is_ok
        assert not HttpResponse(403, "").is_ok
        assert not HttpResponse(500, "").is_ok
