from index_option_brain.data.adapters.mock import SimulatorDataAdapter
from index_option_brain.state.market_state_builder import MarketStateBuilder


async def test_builder_assembles_a_complete_market_state():
    adapter = SimulatorDataAdapter(seed=1)
    builder = MarketStateBuilder(
        index_adapter=adapter, constituent_adapter=adapter, options_adapter=adapter
    )
    expiries = await adapter.get_available_expiries("NIFTY")

    state = await builder.build("NIFTY", expiries[0])

    assert state.index_state.quote.symbol == "NIFTY"
    assert len(state.constituent_state.quotes) > 0
    assert len(state.constituent_state.weights) == len(state.constituent_state.quotes)
    assert len(state.options_state.chain) > 0
