"""Spec §7. OI must NEVER be treated as a standalone BUY/SELL signal.

This brain reports positioning *structure* — where size sits, where the
walls are, where gamma is concentrated, how expensive and how liquid the
chain is. `oi_structure_score` is signed, but nothing here converts it into
a trade: the Signal Engine may only use it as one corroborating domain, and
`DeterministicSignalEngine` will not fire on options evidence alone.

Interpretation used throughout: growth in call open interest above spot is
supply being written into rallies (resistance, bearish-leaning), growth in
put open interest below spot is downside being written (support,
bullish-leaning). Unwinding reverses the reading. This is positioning, not
prophecy — hence "pressure", not "signal".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from decimal import Decimal
from typing import Any

from index_option_brain.brain import indicators as ind
from index_option_brain.brain.config import OptionsBrainConfig
from index_option_brain.contracts.analysis import OptionsAnalysis
from index_option_brain.contracts.enums import OptionType
from index_option_brain.contracts.instruments import OptionQuote
from index_option_brain.contracts.market_state import MarketState


class OptionsBrain(ABC):
    @abstractmethod
    def analyze(self, state: MarketState) -> OptionsAnalysis: ...


class DeterministicOptionsBrain(OptionsBrain):
    def __init__(self, config: OptionsBrainConfig | None = None) -> None:
        self._config = config or OptionsBrainConfig()

    def analyze(self, state: MarketState) -> OptionsAnalysis:
        cfg = self._config
        chain = state.options_state.chain
        spot = float(state.spot)

        by_strike = self._group_by_strike(chain)
        strikes = sorted(by_strike)

        if len(strikes) < cfg.min_strikes:
            return OptionsAnalysis(
                call_pressure=0.0,
                put_pressure=0.0,
                oi_structure_score=0.0,
                iv_score=0.0,
                liquidity_score=0.0,
                confidence=0.0,
                evidence=[
                    (
                        f"Option chain has {len(strikes)} strikes, "
                        f"below the {cfg.min_strikes} needed for structure analysis"
                    )
                ],
                chain_completeness=self._completeness(by_strike, strikes),
            )

        evidence: list[str] = []
        atm_strike = min(strikes, key=lambda s: abs(s - spot))
        atm_index = strikes.index(atm_strike)
        window_strikes = strikes[
            max(0, atm_index - cfg.atm_window) : atm_index + cfg.atm_window + 1
        ]
        evidence.append(f"ATM strike {atm_strike:.0f} against spot {spot:.2f}")

        call_pressure, put_pressure, structure_score, pressure_evidence = self._pressure(
            by_strike, window_strikes, spot, cfg
        )
        evidence.extend(pressure_evidence)

        basis_fields = self._basis(state, cfg)
        if (basis_note := basis_fields.pop("_evidence", None)) is not None:
            evidence.append(basis_note)

        call_walls = self._walls(by_strike, strikes, OptionType.CE, spot, cfg)
        put_walls = self._walls(by_strike, strikes, OptionType.PE, spot, cfg)
        if call_walls:
            evidence.append(
                "Call walls at " + ", ".join(f"{w:.0f}" for w in call_walls)
            )
        if put_walls:
            evidence.append("Put walls at " + ", ".join(f"{w:.0f}" for w in put_walls))

        gamma_zones = self._gamma_zones(by_strike, strikes, cfg)
        if gamma_zones:
            evidence.append(
                "Gamma concentrated at " + ", ".join(f"{z:.0f}" for z in gamma_zones)
            )

        pcr_oi, pcr_volume = self._put_call_ratios(by_strike, window_strikes)
        if pcr_oi is not None:
            evidence.append(f"Put/call OI ratio {pcr_oi:.2f} across the ATM window")

        atm_iv, iv_skew, iv_score, iv_evidence = self._implied_volatility(
            by_strike, strikes, atm_index, cfg
        )
        evidence.extend(iv_evidence)

        liquidity_score, liquidity_evidence = self._liquidity(by_strike, window_strikes, cfg)
        evidence.extend(liquidity_evidence)

        strike_concentration = self._strike_concentration(by_strike, strikes)
        max_pain = self._max_pain(by_strike, strikes)
        completeness = self._completeness(by_strike, strikes)

        confidence = ind.clamp(
            completeness * (0.35 + 0.65 * liquidity_score),
            0.0,
            1.0,
        )

        return OptionsAnalysis(
            call_pressure=ind.clamp(call_pressure, 0.0, 1.0),
            put_pressure=ind.clamp(put_pressure, 0.0, 1.0),
            oi_structure_score=ind.clamp(structure_score),
            iv_score=ind.clamp(iv_score),
            liquidity_score=ind.clamp(liquidity_score, 0.0, 1.0),
            gamma_zones=[Decimal(str(z)) for z in gamma_zones],
            call_walls=[Decimal(str(w)) for w in call_walls],
            put_walls=[Decimal(str(w)) for w in put_walls],
            confidence=confidence,
            evidence=evidence,
            atm_strike=Decimal(str(atm_strike)),
            atm_iv=atm_iv,
            iv_skew=iv_skew,
            pcr_oi=pcr_oi,
            pcr_volume=pcr_volume,
            strike_concentration=strike_concentration,
            max_pain_strike=Decimal(str(max_pain)) if max_pain is not None else None,
            chain_completeness=completeness,
            **basis_fields,
        )

    def _basis(self, state: MarketState, cfg: OptionsBrainConfig) -> dict[str, Any]:
        """Futures positioning, from the forward the chain was priced against.

        Returns nothing at all when the forward was not measured — an
        unsolved basis must not render as a basis of zero, which would read
        as "the futures are flat to carry" when it means "nobody looked".
        """
        options = state.options_state
        excess = options.forward_excess_basis
        if excess is None or options.forward_strikes_used < cfg.basis_min_strikes:
            return {}

        points = float(excess)
        score = ind.clamp(points / cfg.basis_full_scale_points)
        direction = "premium to" if points > 0 else "discount to"
        return {
            "forward_basis": options.forward_basis,
            "excess_basis": excess,
            "basis_score": score,
            "_evidence": (
                f"Forward {options.forward} is {abs(points):.0f} points "
                f"{direction} carry across {options.forward_strikes_used} "
                f"parity strikes ({score:+.2f})"
            ),
        }

    def _group_by_strike(
        self, chain: list[OptionQuote]
    ) -> dict[float, dict[OptionType, OptionQuote]]:
        grouped: dict[float, dict[OptionType, OptionQuote]] = defaultdict(dict)
        for quote in chain:
            grouped[float(quote.contract.strike)][quote.contract.option_type] = quote
        return dict(grouped)

    def _pressure(
        self,
        by_strike: dict[float, dict[OptionType, OptionQuote]],
        window: list[float],
        spot: float,
        cfg: OptionsBrainConfig,
    ) -> tuple[float, float, float, list[str]]:
        """Proximity-weighted OI change on each side of the chain.

        Strikes far from spot are discounted: size parked five strikes out
        constrains price far less than size at the money.
        """
        call_written = 0.0
        put_written = 0.0
        call_unwound = 0.0
        put_unwound = 0.0
        total_oi = 0.0

        for strike in window:
            side = by_strike.get(strike, {})
            distance = abs(strike - spot)
            weight = 1.0 / (1.0 + distance / max(spot * 0.005, 1.0))

            call = side.get(OptionType.CE)
            put = side.get(OptionType.PE)
            if call is not None:
                total_oi += call.open_interest
                change = call.open_interest_change * weight
                if change >= 0:
                    call_written += change
                else:
                    call_unwound += -change
            if put is not None:
                total_oi += put.open_interest
                change = put.open_interest_change * weight
                if change >= 0:
                    put_written += change
                else:
                    put_unwound += -change

        if total_oi <= 0:
            return 0.0, 0.0, 0.0, ["Chain reports no open interest in the ATM window"]

        call_pressure = ind.clamp(
            ind.squash((call_written + put_unwound) / total_oi, cfg.pressure_scale), 0.0, 1.0
        )
        put_pressure = ind.clamp(
            ind.squash((put_written + call_unwound) / total_oi, cfg.pressure_scale), 0.0, 1.0
        )

        net = (put_written + call_unwound - call_written - put_unwound) / total_oi
        structure_score = ind.squash(net, cfg.oi_change_scale)

        evidence = [
            (
                f"OI change in ATM window: calls {call_written - call_unwound:+,.0f}, "
                f"puts {put_written - put_unwound:+,.0f} (proximity weighted)"
            )
        ]
        if abs(structure_score) > 0.3:
            leaning = "supportive" if structure_score > 0 else "resistive"
            evidence.append(
                f"Positioning is {leaning}, but OI alone does not authorize direction"
            )
        return call_pressure, put_pressure, structure_score, evidence

    def _walls(
        self,
        by_strike: dict[float, dict[OptionType, OptionQuote]],
        strikes: list[float],
        option_type: OptionType,
        spot: float,
        cfg: OptionsBrainConfig,
    ) -> list[float]:
        """Open-interest concentrations on the side of spot where they act as
        a barrier: call walls at or above spot, put walls at or below.

        Restricting by side matters. Open interest almost always peaks near
        the money, so an unfiltered "largest OI" search returns the ATM strike
        as both a call wall and a put wall — which is not a wall at all, and
        would make the Strike Engine's wall penalty fire on every candidate.
        """
        sized = [
            (strike, by_strike[strike][option_type].open_interest)
            for strike in strikes
            if option_type in by_strike[strike]
            and by_strike[strike][option_type].open_interest > 0
            and (strike >= spot if option_type is OptionType.CE else strike <= spot)
        ]
        if not sized:
            return []
        sized.sort(key=lambda item: item[1], reverse=True)
        return [strike for strike, _ in sized[: cfg.wall_count]]

    def _gamma_zones(
        self,
        by_strike: dict[float, dict[OptionType, OptionQuote]],
        strikes: list[float],
        cfg: OptionsBrainConfig,
    ) -> list[float]:
        """Strikes carrying the most gamma x open interest — where dealer
        hedging flow is most likely to amplify or pin moves."""
        exposure: list[tuple[float, float]] = []
        for strike in strikes:
            total = 0.0
            for quote in by_strike[strike].values():
                if quote.greeks is None:
                    continue
                total += abs(float(quote.greeks.gamma)) * quote.open_interest
            if total > 0:
                exposure.append((strike, total))
        if not exposure:
            return []
        exposure.sort(key=lambda item: item[1], reverse=True)
        return [strike for strike, _ in exposure[: cfg.gamma_zone_count]]

    def _put_call_ratios(
        self,
        by_strike: dict[float, dict[OptionType, OptionQuote]],
        window: list[float],
    ) -> tuple[float | None, float | None]:
        call_oi = put_oi = call_volume = put_volume = 0
        for strike in window:
            side = by_strike.get(strike, {})
            if (call := side.get(OptionType.CE)) is not None:
                call_oi += call.open_interest
                call_volume += call.volume
            if (put := side.get(OptionType.PE)) is not None:
                put_oi += put.open_interest
                put_volume += put.volume
        pcr_oi = put_oi / call_oi if call_oi > 0 else None
        pcr_volume = put_volume / call_volume if call_volume > 0 else None
        return pcr_oi, pcr_volume

    def _implied_volatility(
        self,
        by_strike: dict[float, dict[OptionType, OptionQuote]],
        strikes: list[float],
        atm_index: int,
        cfg: OptionsBrainConfig,
    ) -> tuple[float | None, float | None, float, list[str]]:
        """ATM IV plus the put-minus-call skew of equidistant wings.

        Skew is the market's price of downside protection relative to upside;
        a steepening put skew is fear being paid for, and it changes which
        structures are worth using even when direction is unchanged.
        """
        evidence: list[str] = []
        atm_strike = strikes[atm_index]
        atm_side = by_strike[atm_strike]
        atm_ivs = [
            float(q.implied_volatility)
            for q in atm_side.values()
            if q.implied_volatility is not None
        ]
        atm_iv = ind.mean(atm_ivs)
        if atm_iv is not None:
            evidence.append(f"ATM implied volatility {atm_iv:.2f}%")

        wing_offset = min(cfg.atm_window, atm_index, len(strikes) - 1 - atm_index)
        iv_skew = None
        if wing_offset > 0:
            otm_put = by_strike[strikes[atm_index - wing_offset]].get(OptionType.PE)
            otm_call = by_strike[strikes[atm_index + wing_offset]].get(OptionType.CE)
            if (
                otm_put is not None
                and otm_call is not None
                and otm_put.implied_volatility is not None
                and otm_call.implied_volatility is not None
            ):
                iv_skew = float(otm_put.implied_volatility) - float(
                    otm_call.implied_volatility
                )
                evidence.append(f"Put-minus-call IV skew {iv_skew:+.2f} points")

        # Reported as a *positioning* score: put skew (fear) reads negative,
        # call skew (chase) positive. Absolute IV level is the Volatility
        # Engine's job, not this brain's.
        iv_score = -ind.squash(iv_skew, cfg.skew_scale) if iv_skew is not None else 0.0
        return atm_iv, iv_skew, iv_score, evidence

    def _liquidity(
        self,
        by_strike: dict[float, dict[OptionType, OptionQuote]],
        window: list[float],
        cfg: OptionsBrainConfig,
    ) -> tuple[float, list[str]]:
        """Liquidity from relative bid-ask spreads near the money.

        Spec §29 requires no options entry on an incomplete chain, and §16
        requires an acceptable spread before any order — both depend on this
        being measured honestly rather than assumed.
        """
        relative_spreads: list[float] = []
        quoted = 0
        total = 0
        for strike in window:
            for quote in by_strike.get(strike, {}).values():
                total += 1
                relative = quote.relative_spread
                if relative is not None:
                    quoted += 1
                    relative_spreads.append(float(relative))

        if not relative_spreads:
            return 0.0, ["No two-sided quotes in the ATM window — treat the chain as illiquid"]

        median_spread = sorted(relative_spreads)[len(relative_spreads) // 2]
        spread_score = ind.clamp(
            1.0 - (median_spread / cfg.max_relative_spread), 0.0, 1.0
        )
        quote_coverage = quoted / total if total else 0.0
        score = spread_score * quote_coverage
        evidence = [
            (
                f"Median relative spread {median_spread * 100:.2f}% "
                f"on {quoted}/{total} quoted contracts near ATM"
            )
        ]
        if score < 0.35:
            evidence.append("Liquidity is poor — slippage risk dominates any edge here")
        return score, evidence

    def _strike_concentration(
        self,
        by_strike: dict[float, dict[OptionType, OptionQuote]],
        strikes: list[float],
    ) -> float:
        totals = [
            float(sum(q.open_interest for q in by_strike[strike].values()))
            for strike in strikes
        ]
        return ind.normalized_hhi(totals) or 0.0

    def _max_pain(
        self,
        by_strike: dict[float, dict[OptionType, OptionQuote]],
        strikes: list[float],
    ) -> float | None:
        """The strike at which the most option value expires worthless.

        Reported as context only — it is a static snapshot statistic, not a
        forecast, and nothing downstream treats it as one.
        """
        if not strikes:
            return None
        pains: list[tuple[float, float]] = []
        for expiry_strike in strikes:
            pain = 0.0
            for strike in strikes:
                side = by_strike[strike]
                if (call := side.get(OptionType.CE)) is not None:
                    pain += max(0.0, expiry_strike - strike) * call.open_interest
                if (put := side.get(OptionType.PE)) is not None:
                    pain += max(0.0, strike - expiry_strike) * put.open_interest
            pains.append((expiry_strike, pain))
        return min(pains, key=lambda item: item[1])[0]

    def _completeness(
        self,
        by_strike: dict[float, dict[OptionType, OptionQuote]],
        strikes: list[float],
    ) -> float:
        """Fraction of expected CE/PE slots actually present, used to gate
        confidence and, upstream, the no-options-entry failure rule."""
        if not strikes:
            return 0.0
        present = sum(len(by_strike[strike]) for strike in strikes)
        return ind.clamp(present / (2 * len(strikes)), 0.0, 1.0)
