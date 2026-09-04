"""Volatility Engine behaviour (spec §8).

The distinction under test is between *level* (where IV sits in its own
history) and *richness* (IV against realized). They are separate fields
because they answer separate questions, and conflating them is how premium
gets sold into a market that is genuinely moving.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from index_option_brain.brain.config import VolatilityBrainConfig
from index_option_brain.brain.volatility_brain import DeterministicVolatilityEngine
from index_option_brain.contracts.enums import IvRegime, OptionType
from index_option_brain.contracts.instruments import (
    IndexQuote,
    OptionContractSpec,
    OptionQuote,
)
from index_option_brain.contracts.market_state import (
    ConstituentState,
    IndexState,
    MarketState,
    OptionsState,
    SectorState,
    VolatilityState,
)

engine = DeterministicVolatilityEngine()


class TestRegimeClassification:
    def test_iv_at_the_top_of_its_history_reads_high(self, state_builder):
        state = state_builder(base_iv=20.0, iv_history=[10.0 + i * 0.1 for i in range(40)])
        analysis = engine.analyze(state)
        assert analysis.regime is IvRegime.HIGH
        assert analysis.iv_percentile == 1.0

    def test_iv_at_the_bottom_of_its_history_reads_low(self, state_builder):
        state = state_builder(base_iv=8.0, iv_history=[15.0 + i * 0.1 for i in range(40)])
        analysis = engine.analyze(state)
        assert analysis.regime is IvRegime.LOW
        # The current print is part of its own history, so the floor is 1/n.
        assert analysis.iv_percentile is not None
        assert analysis.iv_percentile < 0.05

    def test_too_little_history_defaults_to_normal_rather_than_ranking(
        self, uptrend_state: MarketState
    ):
        """One print is not a distribution. Ranking against it would report
        the first observation of a new series as a volatility extreme."""
        thin = uptrend_state.volatility_state.model_copy(update={"atm_iv_history": [14.0]})
        analysis = engine.analyze(uptrend_state.model_copy(update={"volatility_state": thin}))
        assert analysis.regime is IvRegime.NORMAL
        assert analysis.iv_percentile is None
        assert any("too few to rank" in item for item in analysis.evidence)


class TestRichness:
    def test_iv_above_realized_reads_rich(self, state_builder):
        state = state_builder(daily_volatility_pct=0.3, base_iv=22.0, mean_reversion=0.5)
        analysis = engine.analyze(state)
        assert analysis.iv_rv_ratio is not None and analysis.iv_rv_ratio > 1
        assert analysis.iv_score > 0

    def test_iv_below_realized_reads_cheap(self, cheap_volatility_state: MarketState):
        analysis = engine.analyze(cheap_volatility_state)
        assert analysis.iv_rv_ratio is not None and analysis.iv_rv_ratio < 1
        assert analysis.iv_score < 0

    def test_level_and_richness_are_independent(self, cheap_volatility_state: MarketState):
        """High IV is not the same as expensive IV: here IV sits low in its
        own history while realized volatility is higher still."""
        analysis = engine.analyze(cheap_volatility_state)
        assert analysis.regime is IvRegime.LOW
        assert analysis.iv_score < 0
        assert analysis.realized_volatility is not None
        assert analysis.atm_iv is not None
        assert analysis.atm_iv < analysis.realized_volatility


class TestExpectedMove:
    def test_expected_move_is_positive_and_scales_with_time(self, state_builder):
        near = engine.analyze(state_builder(expiry_index=0))
        far = engine.analyze(state_builder(expiry_index=3))
        assert near.expected_move > 0
        assert far.expected_move > near.expected_move

    def test_expected_move_scales_with_implied_volatility(self, state_builder):
        calm = engine.analyze(state_builder(base_iv=10.0))
        stormy = engine.analyze(state_builder(base_iv=30.0))
        assert stormy.expected_move > calm.expected_move

    def test_no_implied_volatility_yields_no_expected_move(self, uptrend_state: MarketState):
        stripped = uptrend_state.volatility_state.model_copy(update={"atm_iv": None})
        blank_chain = uptrend_state.options_state.model_copy(update={"chain": []})
        analysis = engine.analyze(
            uptrend_state.model_copy(
                update={"volatility_state": stripped, "options_state": blank_chain}
            )
        )
        assert analysis.expected_move == 0
        assert analysis.confidence == 0.0


class TestFallbacksAndConfidence:
    def test_atm_iv_falls_back_to_the_chain(self, uptrend_state: MarketState):
        """If the data layer didn't supply an ATM IV, take it from the chain
        rather than assuming a level."""
        stripped = uptrend_state.volatility_state.model_copy(update={"atm_iv": None})
        analysis = engine.analyze(uptrend_state.model_copy(update={"volatility_state": stripped}))
        assert analysis.atm_iv is not None
        assert analysis.atm_iv > 0

    def test_confidence_rises_with_history_length(self, state_builder):
        thin = engine.analyze(state_builder(iv_history=[14.0, 14.1]))
        thick = engine.analyze(state_builder(iv_history=[14.0 + i * 0.05 for i in range(40)]))
        assert thick.confidence > thin.confidence

    def test_expansion_is_reported_with_evidence(self, uptrend_state: MarketState):
        analysis = engine.analyze(uptrend_state)
        assert -1.0 <= analysis.expansion_score <= 1.0
        assert analysis.evidence


# ---------------------------------------------------------------------------
# Expected move, and the ATM straddle cross-check.

_NOW = datetime(2026, 9, 2, 6, 30, tzinfo=UTC)
_EXPIRY = date(2026, 9, 8)


def _leg(
    strike: Decimal,
    option_type: OptionType,
    price: Decimal,
    *,
    quoted: bool = True,
) -> OptionQuote:
    return OptionQuote(
        contract=OptionContractSpec(
            underlying_symbol="NIFTY",
            expiry=_EXPIRY,
            strike=strike,
            option_type=option_type,
            lot_size=65,
            tick_size=Decimal("0.05"),
        ),
        timestamp=_NOW,
        ltp=price,
        bid=price - Decimal("0.5") if quoted else None,
        ask=price + Decimal("0.5") if quoted else None,
        volume=1_000_000,
        open_interest=50_000,
        open_interest_change=0,
        implied_volatility=Decimal(10),
    )


def build_vol_state(
    *,
    atm_iv: float | None,
    days_to_expiry: float | None,
    spot: str = "23900",
    with_atm_straddle: bool = False,
    straddle_scale: float = 1.0,
    drop_put: bool = False,
    quoted: bool = True,
) -> MarketState:
    """A minimal state with a controllable ATM pair.

    Built by hand rather than from the simulator so the straddle can be set
    to a chosen fraction of what IV implies — the whole point being to test
    what happens when the two disagree.
    """
    price = Decimal(spot)
    chain: list[OptionQuote] = []
    if with_atm_straddle and atm_iv is not None and days_to_expiry:
        one_sigma = float(price) * (atm_iv / 100.0) * math.sqrt(days_to_expiry / 365)
        # Split the theoretically correct straddle evenly across the pair,
        # then scale it to create (or not create) a dislocation.
        half = one_sigma * math.sqrt(2 / math.pi) / 2 * straddle_scale
        chain.append(_leg(price, OptionType.CE, Decimal(str(round(half, 2))), quoted=quoted))
        if not drop_put:
            chain.append(
                _leg(price, OptionType.PE, Decimal(str(round(half, 2))), quoted=quoted)
            )

    return MarketState(
        timestamp=_NOW,
        index_state=IndexState(
            quote=IndexQuote(
                symbol="NIFTY",
                timestamp=_NOW,
                ltp=price,
                open=price,
                high=price,
                low=price,
                previous_close=price,
            )
        ),
        constituent_state=ConstituentState(),
        sector_state=SectorState(),
        options_state=OptionsState(chain=chain, expiry=_EXPIRY),
        volatility_state=VolatilityState(
            atm_iv=atm_iv,
            days_to_expiry=days_to_expiry,
            atm_iv_history=[atm_iv] if atm_iv is not None else [],
        ),
    )


def analyse(**kwargs):
    return engine.analyze(build_vol_state(**kwargs))


class TestExpectedMoveAndTheStraddle:
    """"Expected move" names two statistics that differ by a fixed 20%.

    One sigma (`spot x IV x sqrt(T)`) is a containment band; E|move|, which is
    what an ATM straddle is worth, is an average magnitude. They are linked by
    `sqrt(2/pi) = 0.7979` for any spot, IV and tenor — which is what makes the
    observed straddle a free consistency check on the chain, and what makes
    using one as the other a 20% strike-selection error.
    """

    def test_one_sigma_uses_calendar_time(self):
        """Premium decays over the calendar, not the trading session. Using
        252 here would understate weekend risk on a weekly."""
        analysis = analyse(atm_iv=12.0, days_to_expiry=30.0, spot="25000")
        # 25000 * 0.12 * sqrt(30/365)
        assert float(analysis.expected_move) == pytest.approx(860.1, abs=1.0)

    def test_the_absolute_move_is_the_straddle_equivalent(self):
        analysis = analyse(atm_iv=12.0, days_to_expiry=30.0, spot="25000")
        assert analysis.expected_absolute_move is not None
        ratio = float(analysis.expected_absolute_move) / float(analysis.expected_move)
        assert ratio == pytest.approx(math.sqrt(2 / math.pi), abs=1e-4)

    def test_the_two_differ_by_about_a_fifth(self):
        """The number that matters for strike selection: treating the straddle
        figure as a one-sigma band picks strikes 20% too close to spot."""
        analysis = analyse(atm_iv=12.0, days_to_expiry=30.0, spot="25000")
        assert analysis.expected_absolute_move is not None
        shortfall = 1 - float(analysis.expected_absolute_move) / float(
            analysis.expected_move
        )
        assert 0.19 < shortfall < 0.21

    def test_both_are_absent_without_an_iv(self):
        """No IV, no move. A zero would read as a market expected not to
        move."""
        analysis = analyse(atm_iv=None, days_to_expiry=6.0)
        assert analysis.expected_absolute_move is None

    def test_a_coherent_chain_shows_a_small_divergence(self):
        """The check passing is the normal case, and it has to stay quiet."""
        analysis = analyse(atm_iv=10.0, days_to_expiry=6.0, with_atm_straddle=True)
        assert analysis.straddle_divergence is not None
        assert abs(analysis.straddle_divergence) < 0.05
        assert "unreliable" not in " ".join(analysis.evidence)

    def test_the_observed_straddle_is_reported(self):
        analysis = analyse(atm_iv=10.0, days_to_expiry=6.0, with_atm_straddle=True)
        assert analysis.straddle_price is not None
        assert analysis.straddle_price > 0

    def test_a_dislocated_straddle_is_flagged_as_a_data_problem(self):
        """The two numbers are linked by a constant, so they cannot
        legitimately disagree. A gap means a stale IV or an unmarkable book —
        not a market view."""
        analysis = analyse(
            atm_iv=10.0, days_to_expiry=6.0, with_atm_straddle=True, straddle_scale=2.0
        )
        assert analysis.straddle_divergence is not None
        assert analysis.straddle_divergence > 0.5
        joined = " ".join(analysis.evidence)
        assert "unreliable" in joined
        assert "not the model" in joined

    def test_a_one_sided_atm_pair_yields_no_check(self):
        """A measurement or nothing. Half a straddle is not a straddle."""
        analysis = analyse(
            atm_iv=10.0, days_to_expiry=6.0, with_atm_straddle=True, drop_put=True
        )
        assert analysis.straddle_price is None
        assert analysis.straddle_divergence is None

    def test_an_unquoted_atm_pair_yields_no_check(self):
        """`mid` falls back to LTP, and a straddle built from two stale prints
        would manufacture a dislocation out of nothing."""
        analysis = analyse(
            atm_iv=10.0, days_to_expiry=6.0, with_atm_straddle=True, quoted=False
        )
        assert analysis.straddle_divergence is None

    def test_no_chain_yields_no_check(self):
        analysis = analyse(atm_iv=10.0, days_to_expiry=6.0)
        assert analysis.straddle_price is None

    def test_the_tolerance_is_configurable(self):
        strict = DeterministicVolatilityEngine(
            VolatilityBrainConfig(max_straddle_divergence=0.001)
        )
        state = build_vol_state(
            atm_iv=10.0, days_to_expiry=6.0, with_atm_straddle=True, straddle_scale=1.02
        )
        assert "unreliable" in " ".join(strict.analyze(state).evidence)
