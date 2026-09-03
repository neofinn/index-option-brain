"""Market-state engine behaviour (spec §1, §3).

The builder is the only component that knows about individual adapters, and
it is where derived *measurements* (realized volatility, ATM IV, sector
aggregates, session state) are computed. Those are observations, not
judgements — which is why they live here rather than in a brain.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from index_option_brain.contracts.enums import MarketSessionState
from index_option_brain.data.adapters.base import DataAdapterError
from index_option_brain.data.adapters.mock import SimulatorDataAdapter
from index_option_brain.state import InMemoryIvHistoryStore, MarketStateBuilder

AS_OF = datetime(2026, 9, 4, 6, 0, tzinfo=UTC)  # Friday, 11:30 IST


def make_builder(
    adapter: SimulatorDataAdapter, history: InMemoryIvHistoryStore | None = None
) -> MarketStateBuilder:
    return MarketStateBuilder(adapter, adapter, adapter, adapter, history)


@pytest.fixture
def adapter() -> SimulatorDataAdapter:
    return SimulatorDataAdapter(seed=3, as_of=AS_OF)


class TestAssembly:
    async def test_builder_assembles_a_complete_market_state(
        self, adapter: SimulatorDataAdapter
    ):
        state = await make_builder(adapter).build("NIFTY")

        assert state.index_symbol == "NIFTY"
        assert state.index_state.spec is not None
        assert state.index_state.daily_bars
        assert state.index_state.intraday_bars
        assert state.constituent_state.quotes
        assert len(state.constituent_state.weights) == len(state.constituent_state.quotes)
        assert state.constituent_state.sectors
        assert state.options_state.chain
        assert state.options_state.expiry is not None
        assert state.options_state.available_expiries

    async def test_a_specific_expiry_can_be_requested(self, adapter: SimulatorDataAdapter):
        expiries = await adapter.get_available_expiries("NIFTY")
        state = await make_builder(adapter).build("NIFTY", expiries[2])
        assert state.options_state.expiry == expiries[2]
        assert all(q.contract.expiry == expiries[2] for q in state.options_state.chain)

    async def test_the_nearest_expiry_is_the_default(self, adapter: SimulatorDataAdapter):
        expiries = await adapter.get_available_expiries("NIFTY")
        state = await make_builder(adapter).build("NIFTY")
        assert state.options_state.expiry == expiries[0]

    async def test_state_timestamp_comes_from_the_quote(
        self, adapter: SimulatorDataAdapter
    ):
        state = await make_builder(adapter).build("NIFTY")
        assert state.timestamp == state.index_state.quote.timestamp

    async def test_each_state_gets_its_own_identifier(self, adapter: SimulatorDataAdapter):
        builder = make_builder(adapter)
        first = await builder.build("NIFTY")
        second = await builder.build("NIFTY")
        assert first.state_id != second.state_id


class TestDerivedMeasurements:
    async def test_realized_volatility_is_annualized_and_positive(
        self, adapter: SimulatorDataAdapter
    ):
        state = await make_builder(adapter).build("NIFTY")
        realized = state.volatility_state.realized_volatility
        assert realized is not None
        assert 0 < realized < 200

    async def test_realized_volatility_tracks_the_simulated_volatility(self):
        calm = SimulatorDataAdapter(seed=3, as_of=AS_OF, daily_volatility_pct=0.3)
        wild = SimulatorDataAdapter(seed=3, as_of=AS_OF, daily_volatility_pct=2.0)
        calm_state = await make_builder(calm).build("NIFTY")
        wild_state = await make_builder(wild).build("NIFTY")
        assert (
            wild_state.volatility_state.realized_volatility
            > calm_state.volatility_state.realized_volatility
        )

    async def test_atm_iv_averages_both_sides_of_the_nearest_strike(
        self, adapter: SimulatorDataAdapter
    ):
        """Using one side alone would inherit that side's skew."""
        state = await make_builder(adapter).build("NIFTY")
        atm_iv = state.volatility_state.atm_iv
        assert atm_iv is not None

        strikes = {q.contract.strike for q in state.options_state.chain}
        atm_strike = min(strikes, key=lambda s: abs(s - state.spot))
        pair = [
            float(q.implied_volatility)
            for q in state.options_state.chain
            if q.contract.strike == atm_strike and q.implied_volatility is not None
        ]
        assert atm_iv == pytest.approx(sum(pair) / len(pair))

    async def test_days_to_expiry_is_positive_and_shrinks_for_nearer_expiries(
        self, adapter: SimulatorDataAdapter
    ):
        expiries = await adapter.get_available_expiries("NIFTY")
        near = await make_builder(adapter).build("NIFTY", expiries[0])
        far = await make_builder(adapter).build("NIFTY", expiries[3])
        assert near.volatility_state.days_to_expiry is not None
        assert 0 < near.volatility_state.days_to_expiry
        assert far.volatility_state.days_to_expiry > near.volatility_state.days_to_expiry

    async def test_sector_returns_are_weighted_within_each_sector(
        self, adapter: SimulatorDataAdapter
    ):
        state = await make_builder(adapter).build("NIFTY")
        assert state.sector_state.sector_returns
        assert set(state.sector_state.sector_returns) == set(
            state.sector_state.sector_weights
        )
        assert all(weight > 0 for weight in state.sector_state.sector_weights.values())

    async def test_india_vix_is_captured_when_the_adapter_supplies_it(
        self, adapter: SimulatorDataAdapter
    ):
        state = await make_builder(adapter).build("NIFTY")
        assert state.volatility_state.india_vix is not None
        assert state.volatility_state.india_vix_previous_close is not None

    async def test_no_volatility_adapter_leaves_vix_absent_rather_than_guessed(
        self, adapter: SimulatorDataAdapter
    ):
        builder = MarketStateBuilder(adapter, adapter, adapter)
        state = await builder.build("NIFTY")
        assert state.volatility_state.india_vix is None


class TestIvHistory:
    async def test_history_accumulates_across_builds(self, adapter: SimulatorDataAdapter):
        history = InMemoryIvHistoryStore()
        builder = make_builder(adapter, history)
        await builder.build("NIFTY")
        first = len(history.history("NIFTY"))
        await builder.build("NIFTY")
        assert len(history.history("NIFTY")) == first + 1

    async def test_history_is_injected_not_privately_accumulated(
        self, adapter: SimulatorDataAdapter
    ):
        """Spec §36 forbids hidden global state; in BACKTEST/REPLAY the
        history must come from the replayed timeline."""
        seeded = InMemoryIvHistoryStore()
        for value in [11.0, 12.0, 13.0]:
            seeded.record("NIFTY", value)
        state = await make_builder(adapter, seeded).build("NIFTY")
        assert state.volatility_state.atm_iv_history[:3] == [11.0, 12.0, 13.0]

    async def test_history_is_bounded(self):
        store = InMemoryIvHistoryStore(max_observations=5)
        for value in range(20):
            store.record("NIFTY", float(value))
        assert len(store.history("NIFTY")) == 5
        assert store.history("NIFTY")[-1] == 19.0

    async def test_no_history_store_yields_an_empty_history(
        self, adapter: SimulatorDataAdapter
    ):
        state = await make_builder(adapter, None).build("NIFTY")
        assert state.volatility_state.atm_iv_history == []


class TestSessionState:
    @pytest.mark.parametrize(
        "ist_hour,ist_minute,expected",
        [
            (8, 0, MarketSessionState.PRE_MARKET),
            (9, 14, MarketSessionState.PRE_MARKET),
            (9, 20, MarketSessionState.OPENING),
            (11, 30, MarketSessionState.ACTIVE),
            (15, 10, MarketSessionState.CLOSING),
            (16, 0, MarketSessionState.CLOSED),
        ],
    )
    async def test_session_boundaries_follow_ist(
        self,
        adapter: SimulatorDataAdapter,
        ist_hour: int,
        ist_minute: int,
        expected: MarketSessionState,
    ):
        builder = make_builder(adapter)
        moment = datetime(2026, 9, 4, ist_hour, ist_minute, tzinfo=UTC) - timedelta(
            hours=5, minutes=30
        )
        assert builder.session_state(moment) is expected

    async def test_the_built_state_carries_its_session(self, adapter: SimulatorDataAdapter):
        state = await make_builder(adapter).build("NIFTY")
        assert state.session_state is MarketSessionState.ACTIVE


class TestOpeningRange:
    async def test_the_opening_range_brackets_the_early_session(
        self, adapter: SimulatorDataAdapter
    ):
        state = await make_builder(adapter).build("NIFTY")
        opening_range = state.index_state.opening_range
        if opening_range is not None:
            assert opening_range.high >= opening_range.low


class TestBreadthIsNotLoadBearing:
    """Breadth is one of four domains and the only one sourced from a
    session-bounded auction board. Its failure must cost the breadth
    reading, never the state."""

    async def test_a_failing_constituent_feed_still_builds_a_state(
        self, adapter: SimulatorDataAdapter
    ) -> None:
        class _Broken:
            async def get_constituents(self, index_symbol: str):
                raise DataAdapterError("pre-open board unavailable")

            async def get_constituent_quotes(self, symbols: list[str]):
                raise DataAdapterError("pre-open board unavailable")

        builder = MarketStateBuilder(
            adapter, _Broken(), adapter, adapter, InMemoryIvHistoryStore()
        )
        state = await builder.build("NIFTY")

        # The state exists and the other three domains are intact.
        assert state.constituent_state.quotes == []
        assert state.index_state.quote is not None
        assert state.options_state.chain
