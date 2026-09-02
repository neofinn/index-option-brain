"""Spec §8.

Two different questions live here and are kept separate:

  * *Where is IV, historically?* — `regime` and `iv_percentile`, from the
    percentile rank of current ATM IV within its own history.
  * *Is premium rich or cheap?* — `iv_score`, from implied versus realized
    volatility. This is the number the Strategy Engine reads to decide
    between paying premium (debit) and collecting it (credit).

High IV does not mean expensive: IV at the 90th percentile is fair if
realized volatility is running just as hot. Conflating the two is how
option sellers end up short gamma into a genuinely moving market.

A third pair is kept separate for the same reason. "Expected move" names two
different statistics that differ by a fixed 20%:

  * `expected_move` — one standard deviation, `spot x IV x sqrt(T)`. A
    containment band: about 68% of outcomes land inside it.
  * `expected_absolute_move` — `E|move|`, which is what an ATM straddle is
    worth, and exactly `sqrt(2/pi) = 0.7979` times one sigma.

Both are correct answers to different questions, and they are routinely
presented as two ways to compute the same number. Using the straddle figure
as a one-sigma band selects strikes 20% too close to spot.

Because the ratio is a constant for any spot, IV and tenor, the *observed*
straddle gives a free consistency check on the chain. Measured against the
live NIFTY chain it came out at 0.8037 against a theoretical 0.7979 — 0.73%
off. A material gap therefore is not a modelling disagreement; it is a stale
IV, a book too wide to mark, or a real dislocation.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from decimal import Decimal

from index_option_brain.brain import indicators as ind
from index_option_brain.brain.config import VolatilityBrainConfig
from index_option_brain.contracts.analysis import VolatilityAnalysis
from index_option_brain.contracts.enums import IvRegime, OptionType
from index_option_brain.contracts.market_state import MarketState

# E|move| / sigma for a normal distribution. The constant that links the ATM
# straddle price to the one-sigma move, for any spot, IV and tenor.
_E_ABS_OVER_SIGMA = math.sqrt(2.0 / math.pi)


class VolatilityEngine(ABC):
    @abstractmethod
    def analyze(self, state: MarketState) -> VolatilityAnalysis: ...


class DeterministicVolatilityEngine(VolatilityEngine):
    def __init__(self, config: VolatilityBrainConfig | None = None) -> None:
        self._config = config or VolatilityBrainConfig()

    def analyze(self, state: MarketState) -> VolatilityAnalysis:
        cfg = self._config
        volatility_state = state.volatility_state
        spot = float(state.spot)
        evidence: list[str] = []

        atm_iv = volatility_state.atm_iv
        if atm_iv is None:
            atm_iv = self._atm_iv_from_chain(state)
        history = list(volatility_state.atm_iv_history)
        days_to_expiry = volatility_state.days_to_expiry

        rankable = len(history) >= cfg.min_rankable_history
        iv_percentile = (
            ind.percentile_rank(history, atm_iv)
            if atm_iv is not None and rankable
            else None
        )
        regime = self._regime(iv_percentile, cfg)
        if atm_iv is not None:
            if iv_percentile is not None:
                evidence.append(
                    f"ATM IV {atm_iv:.2f}% sits at the {iv_percentile * 100:.0f}th percentile "
                    f"of {len(history)} observations -> {regime.value}"
                )
            else:
                evidence.append(
                    f"ATM IV {atm_iv:.2f}% with only {len(history)} historical observations — "
                    f"too few to rank, reporting {regime.value}"
                )

        realized = volatility_state.realized_volatility
        iv_rv_ratio = None
        iv_score = 0.0
        if atm_iv is not None and realized is not None and realized > 0:
            iv_rv_ratio = atm_iv / realized
            iv_score = ind.squash(iv_rv_ratio - 1.0, cfg.iv_rv_scale)
            richness = "rich" if iv_score > 0 else "cheap"
            evidence.append(
                f"IV/RV {iv_rv_ratio:.2f} (realized {realized:.2f}%) — premium looks {richness}"
            )

        expansion_score, expansion_evidence = self._expansion(
            atm_iv, history, volatility_state, cfg
        )
        evidence.extend(expansion_evidence)

        expected_move = self._expected_move(spot, atm_iv, days_to_expiry, cfg)
        absolute_move = expected_move * _E_ABS_OVER_SIGMA if expected_move > 0 else 0.0
        if expected_move > 0:
            evidence.append(
                f"One-sigma move to expiry {expected_move:.0f} points, average "
                f"absolute move {absolute_move:.0f} points"
                + (f" ({days_to_expiry:.1f} days)" if days_to_expiry is not None else "")
            )

        straddle, divergence, straddle_evidence = self._straddle_check(
            state, spot, expected_move, cfg
        )
        evidence.extend(straddle_evidence)

        confidence = self._confidence(atm_iv, history, realized, cfg)

        return VolatilityAnalysis(
            regime=regime,
            expected_move=Decimal(str(round(expected_move, 2))),
            expected_absolute_move=(
                Decimal(str(round(absolute_move, 2))) if expected_move > 0 else None
            ),
            straddle_price=(
                Decimal(str(round(straddle, 2))) if straddle is not None else None
            ),
            straddle_divergence=divergence,
            iv_score=ind.clamp(iv_score),
            expansion_score=ind.clamp(expansion_score),
            confidence=confidence,
            atm_iv=atm_iv,
            iv_percentile=iv_percentile,
            realized_volatility=realized,
            iv_rv_ratio=iv_rv_ratio,
            days_to_expiry=days_to_expiry,
            evidence=evidence,
        )

    def _atm_iv_from_chain(self, state: MarketState) -> float | None:
        """Fallback when the data layer didn't supply an ATM IV: take it from
        the chain itself rather than guessing a level."""
        chain = state.options_state.chain
        if not chain:
            return None
        spot = float(state.spot)
        atm_strike = min(
            (float(q.contract.strike) for q in chain), key=lambda s: abs(s - spot), default=None
        )
        if atm_strike is None:
            return None
        ivs = [
            float(q.implied_volatility)
            for q in chain
            if q.implied_volatility is not None and float(q.contract.strike) == atm_strike
        ]
        return ind.mean(ivs)

    def _regime(self, iv_percentile: float | None, cfg: VolatilityBrainConfig) -> IvRegime:
        if iv_percentile is None:
            return IvRegime.NORMAL
        if iv_percentile < cfg.low_percentile:
            return IvRegime.LOW
        if iv_percentile < cfg.normal_percentile:
            return IvRegime.NORMAL
        if iv_percentile < cfg.elevated_percentile:
            return IvRegime.ELEVATED
        return IvRegime.HIGH

    def _expansion(
        self,
        atm_iv: float | None,
        history: list[float],
        volatility_state: object,
        cfg: VolatilityBrainConfig,
    ) -> tuple[float, list[str]]:
        """Is volatility being bid up or bled out right now?

        Uses IV against its own recent mean, corroborated by the India VIX
        session change where available.
        """
        evidence: list[str] = []
        components: list[tuple[float | None, float]] = []

        if atm_iv is not None and len(history) >= 2:
            recent_mean = ind.mean(history[-min(len(history), 10) :])
            if recent_mean and recent_mean > 0:
                drift = (atm_iv - recent_mean) / recent_mean
                components.append((ind.squash(drift, cfg.expansion_scale), 0.6))
                if abs(drift) > cfg.expansion_scale:
                    direction = "expanding" if drift > 0 else "contracting"
                    evidence.append(
                        f"ATM IV {drift * 100:+.1f}% vs its recent mean — volatility {direction}"
                    )

        vix = getattr(volatility_state, "india_vix", None)
        vix_previous = getattr(volatility_state, "india_vix_previous_close", None)
        if vix is not None and vix_previous:
            vix_change = (vix - vix_previous) / vix_previous
            components.append((ind.squash(vix_change, cfg.expansion_scale), 0.4))
            evidence.append(f"India VIX {vix:.2f} ({vix_change * 100:+.1f}% on the session)")

        score = ind.blend(*components)
        return (score if score is not None else 0.0), evidence

    def _expected_move(
        self,
        spot: float,
        atm_iv: float | None,
        days_to_expiry: float | None,
        cfg: VolatilityBrainConfig,
    ) -> float:
        """One-sigma move to expiry: spot x IV x sqrt(T).

        Calendar days are used for T because option time decays over the
        calendar, not the trading session — using 252 here would understate
        weekend risk on a Thursday-expiry market.
        """
        if atm_iv is None or days_to_expiry is None or days_to_expiry <= 0:
            return 0.0
        return spot * (atm_iv / 100.0) * math.sqrt(days_to_expiry / cfg.calendar_days_per_year)

    def _straddle_check(
        self,
        state: MarketState,
        spot: float,
        expected_move: float,
        cfg: VolatilityBrainConfig,
    ) -> tuple[float | None, float | None, list[str]]:
        """Compare the observed ATM straddle against what ATM IV implies.

        Returns `(straddle, divergence, evidence)`, all `None`-able, because
        this is a measurement and an unmarkable ATM pair is not one. It reads
        the mid rather than the last trade: a stale LTP on either leg would
        manufacture a dislocation out of nothing.
        """
        chain = state.options_state.chain
        if not chain or expected_move <= 0:
            return None, None, []

        strikes = {quote.contract.strike for quote in chain}
        if not strikes:
            return None, None, []
        atm_strike = min(strikes, key=lambda strike: abs(float(strike) - spot))
        legs = {
            quote.contract.option_type: quote
            for quote in chain
            if quote.contract.strike == atm_strike
        }
        call, put = legs.get(OptionType.CE), legs.get(OptionType.PE)
        if call is None or put is None:
            return None, None, []
        # Both sides have to be genuinely two-sided. `mid` falls back to LTP,
        # and a straddle built from two stale prints is not an observation.
        if None in (call.bid, call.ask, put.bid, put.ask):
            return None, None, []

        straddle = float(call.mid + put.mid)
        if straddle <= 0:
            return None, None, []

        implied = expected_move * _E_ABS_OVER_SIGMA
        divergence = straddle / implied - 1.0
        average_pct = expected_move / spot * 100 * _E_ABS_OVER_SIGMA
        evidence = [
            (
                f"ATM {atm_strike} straddle {straddle:.1f} vs {implied:.1f} "
                f"implied by a {average_pct:.2f}% average move "
                f"({divergence * 100:+.1f}%)"
            )
        ]
        if abs(divergence) > cfg.max_straddle_divergence:
            # Not a view on the market. The two numbers are linked by a
            # constant, so they cannot legitimately disagree by this much.
            evidence.append(
                f"Straddle and ATM IV disagree by {divergence * 100:+.1f}%, beyond "
                f"the {cfg.max_straddle_divergence * 100:.0f}% tolerance — the "
                "chain's IV or its ATM book is unreliable, not the model"
            )
        return straddle, divergence, evidence

    def _confidence(
        self,
        atm_iv: float | None,
        history: list[float],
        realized: float | None,
        cfg: VolatilityBrainConfig,
    ) -> float:
        if atm_iv is None:
            return 0.0
        history_factor = ind.clamp(len(history) / cfg.min_history, 0.0, 1.0)
        realized_factor = 1.0 if realized is not None else 0.6
        return ind.clamp(0.3 + 0.7 * history_factor * realized_factor, 0.0, 1.0)
