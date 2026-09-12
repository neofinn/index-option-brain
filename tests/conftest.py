"""Shared fixtures.

Market snapshots are built through the real `MarketStateBuilder` driven by
the simulator adapter, rather than by hand-assembling a MarketState. That way
the tests exercise the same assembly path production uses, and a contract
change that breaks the builder can't be hidden by fixtures that bypass it.

`as_of` is pinned to a mid-session Wednesday with expiry a week out, so tests
assert on structural reads rather than on whatever the wall clock makes the
session state and days-to-expiry today.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from index_option_brain.contracts.market_state import MarketState
from index_option_brain.data.adapters.mock import SimulatorDataAdapter
from index_option_brain.state import InMemoryIvHistoryStore, MarketStateBuilder

# 2026-09-04 is a Friday; 06:00 UTC is 11:30 IST, mid-session. The nearest
# simulated Thursday expiry is 2026-09-10 — six days out, far enough not to
# trigger the EXPIRY regime and a realistic weekly to trade.
PINNED_NOW = datetime(2026, 9, 4, 6, 0, tzinfo=UTC)

# A Thursday, mid-session: the weekly expires at today's close, which is what
# puts the Regime Engine into EXPIRY.
PINNED_EXPIRY_DAY = datetime(2026, 9, 10, 6, 0, tzinfo=UTC)


def build_state(
    *,
    seed: int = 7,
    iv_history: list[float] | None = None,
    expiry_index: int = 0,
    as_of: datetime = PINNED_NOW,
    **adapter_kwargs: object,
) -> MarketState:
    """Build a MarketState from the simulator with a pinned clock.

    `expiry_index` selects which simulated weekly expiry to load, so a test
    can choose a comfortable ~8-day expiry or a near-dated one deliberately.
    """
    import asyncio

    adapter = SimulatorDataAdapter(seed=seed, as_of=as_of, **adapter_kwargs)  # type: ignore[arg-type]
    history = InMemoryIvHistoryStore()
    default_history = [12.5 + (i % 9) * 0.35 for i in range(40)]
    for value in default_history if iv_history is None else iv_history:
        history.record("NIFTY", value)
    builder = MarketStateBuilder(adapter, adapter, adapter, adapter, history)

    async def _build() -> MarketState:
        expiries = await adapter.get_available_expiries("NIFTY")
        return await builder.build("NIFTY", expiries[expiry_index])

    return asyncio.run(_build())


@pytest.fixture
def state_builder() -> Callable[..., MarketState]:
    return build_state


@pytest.fixture
def uptrend_state() -> MarketState:
    """A broad, participated uptrend: index rising with most names up.

    Intraday drift is set well above one session's worth of noise so the
    session's direction is unambiguous — a drift inside the noise band would
    make the fixture's own premise a coin flip.
    """
    return build_state(daily_drift_pct=0.35, intraday_drift_pct=2.0, breadth_bias=0.6)


@pytest.fixture
def downtrend_state() -> MarketState:
    return build_state(daily_drift_pct=-0.35, intraday_drift_pct=-2.0, breadth_bias=-0.6)


@pytest.fixture
def range_state() -> MarketState:
    """A genuine range: mean-reverting, low volatility, no drift.

    Mean reversion is required — a zero-drift random walk still wanders into
    multi-day trends, so "drift = 0" alone does not simulate a range.
    """
    return build_state(
        daily_drift_pct=0.0,
        daily_volatility_pct=0.35,
        intraday_drift_pct=0.0,
        mean_reversion=0.65,
    )


@pytest.fixture
def narrow_rally_state() -> MarketState:
    """Index up on its heavyweights while most constituents fall — the move
    the Constituent Brain exists to distinguish from a broad rally.

    `heavyweight_bias` pushes the three largest weights up and everything
    else down, which is what makes a rising index and negative breadth
    internally consistent rather than contradictory data.
    """
    return build_state(
        daily_drift_pct=0.3, intraday_drift_pct=2.0, heavyweight_bias=2.6
    )


@pytest.fixture
def rich_volatility_state() -> MarketState:
    """IV above realized, so premium is dear and credit structures fit.

    The calm tape is explicit rather than inherited. These tests used to run
    on `uptrend_state`, whose richness came entirely from how realized
    volatility was being measured: close-to-close over the whole ninety-bar
    window, against the implied volatility of an expiry days away. Measuring
    realized over a window matched to the tenor — and with an estimator that
    accounts for the overnight gap — showed the same fixture to be *cheap*,
    not rich.

    So the premise is now built into the data: a quiet 0.35% daily tape
    under a 14% implied surface, which is a genuine volatility risk premium
    rather than an artifact of a horizon mismatch.
    """
    return build_state(
        daily_drift_pct=0.35,
        intraday_drift_pct=2.0,
        breadth_bias=0.6,
        daily_volatility_pct=0.35,
    )


@pytest.fixture
def cheap_volatility_state() -> MarketState:
    """IV below realized, so premium is cheap and debit structures fit."""
    return build_state(
        daily_drift_pct=0.35,
        intraday_drift_pct=2.0,
        breadth_bias=0.6,
        daily_volatility_pct=1.6,
        base_iv=9.0,
        iv_history=[9.0 + (i % 5) * 0.2 for i in range(40)],
    )


@pytest.fixture
def expiry_day_state() -> MarketState:
    """Expiry day itself — hours to expiry, not days."""
    return build_state(
        daily_drift_pct=0.3, intraday_drift_pct=2.0, as_of=PINNED_EXPIRY_DAY
    )


@pytest.fixture
def spot_price() -> Decimal:
    return Decimal("24500.00")
