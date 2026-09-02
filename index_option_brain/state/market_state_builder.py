"""The market-state engine (spec §1, §3): assembles adapter output into a
single, immutable MarketState. This is the only place that is allowed to
know about individual data adapters — everything downstream (event engine,
brains, risk, execution) consumes MarketState and nothing else.
"""

from __future__ import annotations

from datetime import date

from index_option_brain.contracts.market_state import (
    ConstituentState,
    IndexState,
    MarketState,
    OptionsState,
    SectorState,
    VolatilityState,
)
from index_option_brain.data.adapters.base import (
    ConstituentDataAdapter,
    IndexDataAdapter,
    OptionsChainAdapter,
)


class MarketStateBuilder:
    def __init__(
        self,
        index_adapter: IndexDataAdapter,
        constituent_adapter: ConstituentDataAdapter,
        options_adapter: OptionsChainAdapter,
    ) -> None:
        self._index_adapter = index_adapter
        self._constituent_adapter = constituent_adapter
        self._options_adapter = options_adapter

    async def build(self, index_symbol: str, options_expiry: date) -> MarketState:
        index_quote = await self._index_adapter.get_index_quote(index_symbol)
        constituent_specs = await self._constituent_adapter.get_constituents(index_symbol)
        constituent_quotes = await self._constituent_adapter.get_constituent_quotes(
            [spec.symbol for spec in constituent_specs]
        )
        option_chain = await self._options_adapter.get_option_chain(index_symbol, options_expiry)

        weights = {spec.symbol: float(spec.weight) for spec in constituent_specs}

        return MarketState(
            timestamp=index_quote.timestamp,
            index_state=IndexState(quote=index_quote),
            constituent_state=ConstituentState(quotes=constituent_quotes, weights=weights),
            sector_state=SectorState(),
            options_state=OptionsState(chain=option_chain),
            volatility_state=VolatilityState(),
        )
