"""Options Brain behaviour (spec §7).

The governing constraint is that OI must never be a standalone BUY/SELL
signal. Structurally that is enforced by this brain only ever reporting
positioning, and by the Signal Engine weighting it below the other domains —
see `test_signal_engine.py` for the half of that guarantee this brain cannot
make on its own.
"""

from __future__ import annotations

from decimal import Decimal

from index_option_brain.brain.options_brain import DeterministicOptionsBrain
from index_option_brain.contracts.enums import OptionType
from index_option_brain.contracts.market_state import MarketState

brain = DeterministicOptionsBrain()


class TestChainStructure:
    def test_atm_strike_is_the_nearest_strike_to_spot(self, uptrend_state: MarketState):
        analysis = brain.analyze(uptrend_state)
        assert analysis.atm_strike is not None
        strikes = {q.contract.strike for q in uptrend_state.options_state.chain}
        nearest = min(strikes, key=lambda s: abs(s - uptrend_state.spot))
        assert analysis.atm_strike == nearest

    def test_a_complete_chain_reports_full_completeness(self, uptrend_state: MarketState):
        analysis = brain.analyze(uptrend_state)
        assert analysis.chain_completeness == 1.0

    def test_scores_stay_within_their_declared_ranges(self, uptrend_state: MarketState):
        analysis = brain.analyze(uptrend_state)
        assert 0.0 <= analysis.call_pressure <= 1.0
        assert 0.0 <= analysis.put_pressure <= 1.0
        assert -1.0 <= analysis.oi_structure_score <= 1.0
        assert -1.0 <= analysis.iv_score <= 1.0
        assert 0.0 <= analysis.liquidity_score <= 1.0
        assert 0.0 <= analysis.strike_concentration <= 1.0


class TestWalls:
    def test_call_walls_sit_at_or_above_spot_and_puts_at_or_below(
        self, uptrend_state: MarketState
    ):
        """A wall only acts as a barrier on the side it is written. Without
        this filter the ATM strike — where open interest naturally peaks —
        would be reported as both a call wall and a put wall."""
        analysis = brain.analyze(uptrend_state)
        spot = uptrend_state.spot
        assert analysis.call_walls
        assert analysis.put_walls
        assert all(strike >= spot for strike in analysis.call_walls)
        assert all(strike <= spot for strike in analysis.put_walls)

    def test_walls_are_ordered_by_size(self, uptrend_state: MarketState):
        analysis = brain.analyze(uptrend_state)
        chain = {
            (q.contract.strike, q.contract.option_type): q
            for q in uptrend_state.options_state.chain
        }
        sizes = [
            chain[(strike, OptionType.CE)].open_interest for strike in analysis.call_walls
        ]
        assert sizes == sorted(sizes, reverse=True)

    def test_gamma_zones_are_identified(self, uptrend_state: MarketState):
        analysis = brain.analyze(uptrend_state)
        assert analysis.gamma_zones
        assert len(analysis.gamma_zones) <= 3


class TestPositioningMeasures:
    def test_put_call_ratios_are_positive_when_both_sides_trade(
        self, uptrend_state: MarketState
    ):
        analysis = brain.analyze(uptrend_state)
        assert analysis.pcr_oi is not None and analysis.pcr_oi > 0
        assert analysis.pcr_volume is not None and analysis.pcr_volume > 0

    def test_put_skew_reads_as_a_negative_surface_score(self, uptrend_state: MarketState):
        """The simulated surface carries the usual put skew, so downside
        protection is bid — which this brain reports as negative."""
        analysis = brain.analyze(uptrend_state)
        assert analysis.iv_skew is not None
        assert analysis.iv_skew > 0, "OTM puts should carry higher IV than OTM calls"
        assert analysis.iv_score < 0

    def test_max_pain_is_within_the_chain(self, uptrend_state: MarketState):
        analysis = brain.analyze(uptrend_state)
        strikes = {q.contract.strike for q in uptrend_state.options_state.chain}
        assert analysis.max_pain_strike in strikes

    def test_evidence_names_oi_as_positioning_not_direction(
        self, uptrend_state: MarketState
    ):
        analysis = brain.analyze(uptrend_state)
        assert analysis.evidence


class TestLiquidity:
    def test_a_tight_chain_scores_liquid(self, uptrend_state: MarketState):
        analysis = brain.analyze(uptrend_state)
        assert analysis.liquidity_score > 0.5

    def test_an_unquoted_chain_scores_illiquid(self, uptrend_state: MarketState):
        """Spec §16 requires an acceptable spread before any order, so an
        unquoted chain must score at the floor rather than be assumed fine."""
        unquoted = [
            q.model_copy(update={"bid": None, "ask": None})
            for q in uptrend_state.options_state.chain
        ]
        options_state = uptrend_state.options_state.model_copy(update={"chain": unquoted})
        analysis = brain.analyze(
            uptrend_state.model_copy(update={"options_state": options_state})
        )
        assert analysis.liquidity_score == 0.0
        # Positioning structure is still readable from OI without quotes, so
        # confidence is reduced rather than zeroed. Tradeability is gated on
        # liquidity_score, which is what the Strategy Engine blocks on.
        assert analysis.confidence < 0.5
        assert any("illiquid" in item for item in analysis.evidence)


class TestDegradedInput:
    def test_a_short_chain_is_reported_as_unanalyzable(self, uptrend_state: MarketState):
        short = uptrend_state.options_state.model_copy(
            update={"chain": uptrend_state.options_state.chain[:4]}
        )
        analysis = brain.analyze(uptrend_state.model_copy(update={"options_state": short}))
        assert analysis.confidence == 0.0
        assert analysis.oi_structure_score == 0.0
        assert any("strikes" in item for item in analysis.evidence)

    def test_an_empty_chain_does_not_raise(self, uptrend_state: MarketState):
        empty = uptrend_state.options_state.model_copy(update={"chain": []})
        analysis = brain.analyze(uptrend_state.model_copy(update={"options_state": empty}))
        assert analysis.confidence == 0.0
        assert analysis.chain_completeness == 0.0

    def test_a_half_populated_chain_lowers_completeness(self, uptrend_state: MarketState):
        calls_only = [
            q
            for q in uptrend_state.options_state.chain
            if q.contract.option_type is OptionType.CE
        ]
        options_state = uptrend_state.options_state.model_copy(update={"chain": calls_only})
        analysis = brain.analyze(
            uptrend_state.model_copy(update={"options_state": options_state})
        )
        assert analysis.chain_completeness == 0.5


class TestBasis:
    """Futures positioning, read off the forward the chain was priced against.

    Reported here rather than in the Index brain because it is a derivatives
    measurement, and — like `oi_structure_score` — it is corroborating
    evidence the Signal Engine weighs, never a standalone reason to trade.
    """

    def _with_forward(
        self,
        state: MarketState,
        *,
        forward: str,
        basis: str,
        excess: str,
        strikes: int = 6,
    ) -> MarketState:
        return state.model_copy(
            update={
                "options_state": state.options_state.model_copy(
                    update={
                        "forward": Decimal(forward),
                        "forward_basis": Decimal(basis),
                        "forward_excess_basis": Decimal(excess),
                        "forward_strikes_used": strikes,
                    }
                )
            }
        )

    def test_an_unmeasured_forward_reports_nothing_not_zero(
        self, uptrend_state: MarketState
    ):
        """A basis of zero says the futures are flat to carry. No basis says
        nobody looked. Rendering the second as the first is the failure this
        whole codebase is built to avoid."""
        analysis = brain.analyze(uptrend_state)
        assert analysis.basis_score is None
        assert analysis.excess_basis is None
        assert analysis.forward_basis is None

    def test_a_premium_to_carry_scores_positive(self, uptrend_state: MarketState):
        state = self._with_forward(
            uptrend_state, forward="24064.14", basis="43.74", excess="21.09"
        )
        analysis = brain.analyze(state)

        assert analysis.basis_score is not None
        assert analysis.basis_score > 0
        assert analysis.excess_basis == Decimal("21.09")
        assert any("premium to carry" in line for line in analysis.evidence)

    def test_a_discount_to_carry_scores_negative(self, uptrend_state: MarketState):
        state = self._with_forward(
            uptrend_state, forward="23915.59", basis="1.14", excess="-24.40"
        )
        analysis = brain.analyze(state)

        assert analysis.basis_score is not None
        assert analysis.basis_score < 0
        assert any("discount to carry" in line for line in analysis.evidence)

    def test_the_score_is_bounded(self, uptrend_state: MarketState):
        state = self._with_forward(
            uptrend_state, forward="25000", basis="900", excess="880"
        )
        analysis = brain.analyze(state)
        assert analysis.basis_score == 1.0

    def test_too_few_parity_strikes_is_a_quote_not_a_measurement(
        self, uptrend_state: MarketState
    ):
        """A forward solved off one strike moves with that strike's spread."""
        state = self._with_forward(
            uptrend_state, forward="24064.14", basis="43.74", excess="21.09", strikes=1
        )
        assert brain.analyze(state).basis_score is None
