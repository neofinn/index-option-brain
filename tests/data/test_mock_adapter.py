"""Simulator adapter behaviour.

Two properties are load-bearing for every brain test: the data is internally
coherent (premiums and greeks come from the same pricing evaluation), and
reads are idempotent (the same snapshot twice returns the same numbers). A
builder that fetched the spot twice and got two different prices would
assemble a MarketState that never existed.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from itertools import pairwise

import pytest

from index_option_brain.contracts.enums import BarInterval, OptionType
from index_option_brain.data.adapters.base import DataAdapterError
from index_option_brain.data.adapters.mock import SimulatorDataAdapter

AS_OF = datetime(2026, 9, 4, 6, 0, tzinfo=UTC)


@pytest.fixture
def adapter() -> SimulatorDataAdapter:
    return SimulatorDataAdapter(seed=7, as_of=AS_OF)


class TestDeterminismAndIdempotency:
    async def test_the_same_seed_reproduces_the_same_market(self):
        a = SimulatorDataAdapter(seed=7, as_of=AS_OF)
        b = SimulatorDataAdapter(seed=7, as_of=AS_OF)
        assert (await a.get_index_quote("NIFTY")).ltp == (await b.get_index_quote("NIFTY")).ltp

    async def test_different_seeds_produce_different_markets(self):
        a = SimulatorDataAdapter(seed=1, as_of=AS_OF)
        b = SimulatorDataAdapter(seed=2, as_of=AS_OF)
        assert (await a.get_index_quote("NIFTY")).ltp != (await b.get_index_quote("NIFTY")).ltp

    async def test_repeated_reads_return_the_same_snapshot(
        self, adapter: SimulatorDataAdapter
    ):
        """Values derive from the seed and the thing being priced, never from
        a mutable stream position."""
        first = await adapter.get_index_quote("NIFTY")
        second = await adapter.get_index_quote("NIFTY")
        assert first == second

        chain_first = await adapter.get_option_chain(
            "NIFTY", (await adapter.get_available_expiries("NIFTY"))[0]
        )
        chain_second = await adapter.get_option_chain(
            "NIFTY", (await adapter.get_available_expiries("NIFTY"))[0]
        )
        assert chain_first == chain_second

    async def test_reads_are_order_independent(self, adapter: SimulatorDataAdapter):
        quote_first = await adapter.get_index_quote("NIFTY")
        await adapter.get_constituent_quotes(["RELIANCE"])
        await adapter.get_india_vix()
        assert await adapter.get_index_quote("NIFTY") == quote_first


class TestIndexData:
    async def test_unknown_symbols_raise_rather_than_inventing_data(
        self, adapter: SimulatorDataAdapter
    ):
        with pytest.raises(DataAdapterError):
            await adapter.get_index_quote("NOT_A_REAL_INDEX")
        with pytest.raises(DataAdapterError):
            await adapter.get_index_spec("NOT_A_REAL_INDEX")

    async def test_the_quote_is_internally_consistent(self, adapter: SimulatorDataAdapter):
        quote = await adapter.get_index_quote("NIFTY")
        assert quote.low <= quote.ltp <= quote.high
        assert quote.low <= quote.open <= quote.high
        assert quote.previous_close > 0

    async def test_daily_bars_are_ordered_and_consistent(
        self, adapter: SimulatorDataAdapter
    ):
        bars = await adapter.get_index_bars("NIFTY", BarInterval.DAY, 30)
        assert len(bars) == 30
        assert [b.timestamp for b in bars] == sorted(b.timestamp for b in bars)
        for bar in bars:
            assert bar.low <= bar.open <= bar.high
            assert bar.low <= bar.close <= bar.high
            assert bar.volume > 0

    async def test_intraday_bars_respect_the_requested_interval(
        self, adapter: SimulatorDataAdapter
    ):
        five = await adapter.get_index_bars("NIFTY", BarInterval.MINUTE_5, 12)
        fifteen = await adapter.get_index_bars("NIFTY", BarInterval.MINUTE_15, 12)
        assert five
        assert fifteen
        gap = (five[1].timestamp - five[0].timestamp).total_seconds() / 60
        assert gap == 5
        gap = (fifteen[1].timestamp - fifteen[0].timestamp).total_seconds() / 60
        assert gap == 15

    async def test_zero_bars_requested_returns_nothing(self, adapter: SimulatorDataAdapter):
        assert await adapter.get_index_bars("NIFTY", BarInterval.DAY, 0) == []

    async def test_drift_shapes_the_history_without_moving_the_current_price(self):
        """The path is rebased so strike-relative assertions don't shift with
        the drift setting."""
        up = SimulatorDataAdapter(seed=7, as_of=AS_OF, daily_drift_pct=0.5)
        flat = SimulatorDataAdapter(seed=7, as_of=AS_OF, daily_drift_pct=0.0)
        up_bars = await up.get_index_bars("NIFTY", BarInterval.DAY, 60)
        flat_bars = await flat.get_index_bars("NIFTY", BarInterval.DAY, 60)
        assert up_bars[0].close < flat_bars[0].close
        assert up_bars[-1].close == pytest.approx(flat_bars[-1].close, rel=1e-9)


class TestConstituents:
    async def test_constituents_carry_weights_and_sectors(
        self, adapter: SimulatorDataAdapter
    ):
        specs = await adapter.get_constituents("NIFTY")
        assert specs
        assert all(spec.weight > 0 for spec in specs)
        assert all(spec.sector for spec in specs)
        assert all(spec.index_symbol == "NIFTY" for spec in specs)

    async def test_unknown_index_has_no_constituents(self, adapter: SimulatorDataAdapter):
        with pytest.raises(DataAdapterError):
            await adapter.get_constituents("NOT_A_REAL_INDEX")

    async def test_quotes_are_internally_consistent(self, adapter: SimulatorDataAdapter):
        quotes = await adapter.get_constituent_quotes(["RELIANCE", "INFY"])
        assert len(quotes) == 2
        for quote in quotes:
            assert quote.low <= quote.ltp <= quote.high
            assert quote.previous_close > 0
            assert quote.volume > 0

    async def test_heavyweight_bias_splits_the_cross_section(self):
        """The only internally consistent way to simulate a narrow rally."""
        narrow = SimulatorDataAdapter(
            seed=7, as_of=AS_OF, intraday_drift_pct=2.0, heavyweight_bias=2.6
        )
        specs = await narrow.get_constituents("NIFTY")
        quotes = await narrow.get_constituent_quotes([s.symbol for s in specs])
        rising = [q for q in quotes if q.change_pct > 0]
        falling = [q for q in quotes if q.change_pct < 0]
        assert rising and falling
        assert len(falling) > len(rising)


class TestOptionChain:
    async def test_expiries_are_weekly_and_ordered(self, adapter: SimulatorDataAdapter):
        expiries = await adapter.get_available_expiries("NIFTY")
        assert expiries == sorted(expiries)
        assert all(e.weekday() == 3 for e in expiries), "simulated weeklies expire Thursday"
        gaps = {(b - a).days for a, b in pairwise(expiries)}
        assert gaps == {7}

    async def test_expiry_day_keeps_todays_contract_available(self):
        """The weekly is live until the close, so it must not jump a week."""
        thursday = datetime(2026, 9, 10, 6, 0, tzinfo=UTC)
        adapter = SimulatorDataAdapter(seed=7, as_of=thursday)
        expiries = await adapter.get_available_expiries("NIFTY")
        assert expiries[0] == thursday.date()

    async def test_the_chain_spans_both_sides_of_spot(self, adapter: SimulatorDataAdapter):
        expiries = await adapter.get_available_expiries("NIFTY")
        chain = await adapter.get_option_chain("NIFTY", expiries[0])
        spot = (await adapter.get_index_quote("NIFTY")).ltp
        strikes = sorted({q.contract.strike for q in chain})
        assert min(strikes) < spot < max(strikes)
        assert {q.contract.option_type for q in chain} == {OptionType.CE, OptionType.PE}

    async def test_every_quote_is_two_sided_and_bracketed(
        self, adapter: SimulatorDataAdapter
    ):
        expiries = await adapter.get_available_expiries("NIFTY")
        chain = await adapter.get_option_chain("NIFTY", expiries[0])
        for quote in chain:
            assert quote.ltp > Decimal(0)
            assert quote.bid is not None and quote.ask is not None
            assert quote.bid <= quote.ltp <= quote.ask
            assert quote.open_interest > 0

    async def test_greeks_are_coherent_with_the_option_type(
        self, adapter: SimulatorDataAdapter
    ):
        """Premiums and greeks come from the same pricing evaluation, so a
        call's delta is positive and a put's negative — random greeks would
        let a brain pass tests while being arithmetically incoherent."""
        expiries = await adapter.get_available_expiries("NIFTY")
        chain = await adapter.get_option_chain("NIFTY", expiries[0])
        spot = (await adapter.get_index_quote("NIFTY")).ltp
        for quote in chain:
            assert quote.greeks is not None
            if quote.contract.option_type is OptionType.CE:
                assert 0 <= quote.greeks.delta <= 1
                assert quote.greeks.theta <= 0
            else:
                assert -1 <= quote.greeks.delta <= 0
                # A deep in-the-money European put can carry *positive*
                # theta: its value includes the discounted strike, and that
                # discount unwinds toward par as expiry approaches. Indian
                # index options are European, so this is correct, not a bug.
                deep_itm = quote.contract.strike > spot * Decimal("1.02")
                if not deep_itm:
                    assert quote.greeks.theta <= 0
            assert quote.greeks.gamma >= 0
            assert quote.greeks.vega >= 0

    async def test_at_the_money_options_decay(self, adapter: SimulatorDataAdapter):
        """The case that actually matters for a premium seller: ATM options on
        both sides must lose value with time."""
        expiries = await adapter.get_available_expiries("NIFTY")
        chain = await adapter.get_option_chain("NIFTY", expiries[0])
        spot = (await adapter.get_index_quote("NIFTY")).ltp
        atm_strike = min(
            {q.contract.strike for q in chain}, key=lambda s: abs(s - spot)
        )
        atm = [q for q in chain if q.contract.strike == atm_strike]
        assert len(atm) == 2
        for quote in atm:
            assert quote.greeks is not None
            assert quote.greeks.theta < 0

    async def test_intrinsic_value_is_respected(self, adapter: SimulatorDataAdapter):
        expiries = await adapter.get_available_expiries("NIFTY")
        chain = await adapter.get_option_chain("NIFTY", expiries[0])
        spot = (await adapter.get_index_quote("NIFTY")).ltp
        for quote in chain:
            if quote.contract.option_type is OptionType.CE:
                intrinsic = max(Decimal(0), spot - quote.contract.strike)
            else:
                intrinsic = max(Decimal(0), quote.contract.strike - spot)
            assert quote.ltp >= intrinsic * Decimal("0.95")

    async def test_the_surface_carries_a_put_skew(self, adapter: SimulatorDataAdapter):
        expiries = await adapter.get_available_expiries("NIFTY")
        chain = await adapter.get_option_chain("NIFTY", expiries[0])
        spot = float((await adapter.get_index_quote("NIFTY")).ltp)
        strikes = sorted({float(q.contract.strike) for q in chain})
        atm = min(strikes, key=lambda s: abs(s - spot))
        offset = 5 * 50
        by_key = {(float(q.contract.strike), q.contract.option_type): q for q in chain}
        otm_put = by_key.get((atm - offset, OptionType.PE))
        otm_call = by_key.get((atm + offset, OptionType.CE))
        assert otm_put is not None and otm_call is not None
        assert otm_put.implied_volatility > otm_call.implied_volatility

    async def test_spreads_widen_away_from_the_money(self, adapter: SimulatorDataAdapter):
        expiries = await adapter.get_available_expiries("NIFTY")
        chain = await adapter.get_option_chain("NIFTY", expiries[0])
        spot = (await adapter.get_index_quote("NIFTY")).ltp
        calls = [q for q in chain if q.contract.option_type is OptionType.CE]
        atm = min(calls, key=lambda q: abs(q.contract.strike - spot))
        far = max(calls, key=lambda q: abs(q.contract.strike - spot))
        assert far.relative_spread > atm.relative_spread


class TestOtherFeeds:
    async def test_india_vix_returns_current_and_previous(
        self, adapter: SimulatorDataAdapter
    ):
        current, previous = await adapter.get_india_vix()
        assert current > 0
        assert previous > 0

    async def test_account_snapshot_has_non_negative_margin(
        self, adapter: SimulatorDataAdapter
    ):
        snapshot = await adapter.get_account_snapshot()
        assert snapshot.available_margin >= Decimal(0)
        assert snapshot.net_equity >= Decimal(0)
