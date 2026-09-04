"""Event detection and the significance filter (spec §4).

Two properties matter more than coverage of every trigger.

**Quiet when the market is quiet.** An engine that fires on every tick is
worse than a timer: it costs the same and hides the signal. A large share of
these tests assert that something does *not* fire.

**Never fires on unmeasured data.** A detector treating a missing India VIX
as zero would report a volatility collapse every time the feed dropped a
field. The absence of a measurement is not a measurement.
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest

from index_option_brain.contracts.enums import (
    MarketSessionState,
    OptionType,
    TriggerType,
)
from index_option_brain.contracts.events import Event
from index_option_brain.contracts.market_state import MarketState, OpeningRange
from index_option_brain.contracts.risk import ScheduledEvent
from index_option_brain.events import (
    DeterministicTriggerEngine,
    ScheduledEventCalendar,
    SignificanceFilterConfig,
    ThresholdSignificanceFilter,
    TriggerEngineConfig,
)
from index_option_brain.events.detectors import CALENDAR_ONLY_TRIGGERS
from tests.events.conftest import (
    NOW,
    analysis,
    bar,
    constituent,
    default_chain,
    option,
    state,
)


def fired(previous: MarketState | None, current: MarketState, **cfg) -> set[TriggerType]:
    engine = DeterministicTriggerEngine(TriggerEngineConfig(**cfg))
    return {event.trigger_type for event in engine.detect(previous, current)}


def events_of(
    previous: MarketState | None, current: MarketState, trigger: TriggerType, **cfg
) -> list[Event]:
    engine = DeterministicTriggerEngine(TriggerEngineConfig(**cfg))
    return [
        event for event in engine.detect(previous, current) if event.trigger_type is trigger
    ]


class TestTheFirstTick:
    def test_nothing_comparative_fires_without_a_previous_state(self):
        """A system reporting "significant price movement" the moment it
        starts is reporting its own startup."""
        triggers = fired(None, state(ltp="24500", previous_close="23900"))
        assert TriggerType.SIGNIFICANT_PRICE_MOVEMENT not in triggers
        assert TriggerType.BREAKOUT not in triggers
        assert TriggerType.LARGE_OI_ADDITION not in triggers

    def test_a_session_boundary_still_fires_on_the_first_tick(self):
        """The one exception. A session state is a fact about the clock the
        snapshot carries, and a process starting at 09:15 needs to know the
        market just opened."""
        triggers = fired(None, state(session=MarketSessionState.OPENING))
        assert TriggerType.MARKET_OPEN in triggers

    def test_an_exceptional_move_fires_on_the_first_tick(self):
        """Measured against the previous close, not the last snapshot: what
        makes a day exceptional is where it has got to."""
        triggers = fired(None, state(ltp="24600", previous_close="23900"))
        assert TriggerType.EXCEPTIONAL_MARKET_EVENT in triggers


class TestPriceMovement:
    def test_a_large_move_fires(self):
        before = state(with_analysis=analysis(atr="150"))
        assert TriggerType.SIGNIFICANT_PRICE_MOVEMENT in fired(
            before, state(ltp="24010")
        )

    def test_a_small_move_does_not(self):
        before = state(with_analysis=analysis(atr="150"))
        assert TriggerType.SIGNIFICANT_PRICE_MOVEMENT not in fired(
            before, state(ltp="23905")
        )

    def test_the_threshold_is_atr_relative_when_atr_is_known(self):
        """A fixed number of index points means something different at 12%
        volatility than at 30%."""
        before = state(with_analysis=analysis(atr="150"))
        events = events_of(before, state(ltp="24010"), TriggerType.SIGNIFICANT_PRICE_MOVEMENT)
        assert events[0].payload["basis"] == "atr"

    def test_it_falls_back_to_percent_without_atr(self):
        """Which is the situation until bars exist."""
        events = events_of(
            state(), state(ltp="24010"), TriggerType.SIGNIFICANT_PRICE_MOVEMENT
        )
        assert events[0].payload["basis"] == "pct"

    def test_significance_scales_with_size(self):
        """"Just over the line" and "enormous" must not score the same — the
        filter's floor is all that stands between a quiet tick and a full
        analysis."""
        before = state(with_analysis=analysis(atr="150"))
        small = events_of(before, state(ltp="23985"), TriggerType.SIGNIFICANT_PRICE_MOVEMENT)
        large = events_of(before, state(ltp="24400"), TriggerType.SIGNIFICANT_PRICE_MOVEMENT)
        assert small and large
        assert small[0].significance_score < large[0].significance_score

    def test_significance_never_exceeds_one(self):
        before = state(with_analysis=analysis(atr="150"))
        events = events_of(before, state(ltp="30000"), TriggerType.SIGNIFICANT_PRICE_MOVEMENT)
        assert events[0].significance_score == 1.0


class TestBreakout:
    def test_crossing_the_previous_session_high_fires(self):
        before = state(ltp="23890", daily_bars=[bar("23900")])
        assert TriggerType.BREAKOUT in fired(before, state(ltp="23950", daily_bars=[bar("23900")]))

    def test_crossing_the_previous_session_low_fires_a_breakdown(self):
        before = state(ltp="23890", daily_bars=[bar("23900")])
        assert TriggerType.BREAKDOWN in fired(
            before, state(ltp="23860", daily_bars=[bar("23900")])
        )

    def test_staying_above_the_level_does_not_re_fire(self):
        """The crossing is the event. Re-announcing it every tick while price
        sits above the level would drown everything else."""
        above = state(ltp="23950", daily_bars=[bar("23900")])
        assert TriggerType.BREAKOUT not in fired(
            above, state(ltp="23960", daily_bars=[bar("23900")])
        )

    def test_no_bars_means_no_breakout(self):
        before = state(ltp="23890", daily_bars=[])
        assert TriggerType.BREAKOUT not in fired(
            before, state(ltp="24500", daily_bars=[])
        )


class TestOpeningRange:
    def test_completion_fires_once(self):
        opening = OpeningRange(high=Decimal(23950), low=Decimal(23850), completed=True)
        forming = OpeningRange(high=Decimal(23950), low=Decimal(23850), completed=False)
        assert TriggerType.OPENING_RANGE_COMPLETION in fired(
            state(opening_range=forming), state(opening_range=opening)
        )
        assert TriggerType.OPENING_RANGE_COMPLETION not in fired(
            state(opening_range=opening), state(opening_range=opening)
        )

    def test_a_break_of_the_range_fires(self):
        opening = OpeningRange(high=Decimal(23950), low=Decimal(23850), completed=True)
        before = state(ltp="23900", opening_range=opening)
        events = events_of(
            before, state(ltp="23980", opening_range=opening), TriggerType.OPENING_RANGE_EVENT
        )
        assert events and events[0].payload["direction"] == "above"

    def test_an_incomplete_range_cannot_be_broken(self):
        forming = OpeningRange(high=Decimal(23950), low=Decimal(23850), completed=False)
        before = state(ltp="23900", opening_range=forming)
        assert TriggerType.OPENING_RANGE_EVENT not in fired(
            before, state(ltp="23980", opening_range=forming)
        )


class TestVwapCrossing:
    def test_a_crossing_fires_when_vwap_is_published(self):
        before = state(ltp="23890", vwap="23900")
        assert TriggerType.VWAP_CROSSING in fired(
            before, state(ltp="23910", vwap="23900")
        )

    def test_it_stays_silent_when_vwap_is_absent(self):
        """NSE's public feed does not publish an index VWAP, and the detector
        must not cross a fabricated line."""
        before = state(ltp="23000", vwap=None)
        assert TriggerType.VWAP_CROSSING not in fired(
            before, state(ltp="24800", vwap=None)
        )


class TestLevelTest:
    def test_approaching_a_level_the_last_analysis_found_fires(self):
        """Levels come from the previous cycle's analysis, which is the only
        source available — detection runs before analysis. It also means a
        level is only tested once it has been reasoned about."""
        before = state(ltp="23700", with_analysis=analysis(atr="150", support=["23500"]))
        events = events_of(before, state(ltp="23515"), TriggerType.SUPPORT_RESISTANCE_TEST)
        assert events and events[0].payload["kind"] == "support"

    def test_sitting_at_a_level_is_one_test_not_twenty(self):
        at_level = state(ltp="23510", with_analysis=analysis(atr="150", support=["23500"]))
        assert TriggerType.SUPPORT_RESISTANCE_TEST not in fired(
            at_level, state(ltp="23505")
        )

    def test_no_analysis_means_no_levels_to_test(self):
        assert TriggerType.SUPPORT_RESISTANCE_TEST not in fired(
            state(), state(ltp="23500")
        )

    def test_no_atr_means_no_tolerance_to_measure_against(self):
        before = state(ltp="23700", with_analysis=analysis(atr=None, support=["23500"]))
        assert TriggerType.SUPPORT_RESISTANCE_TEST not in fired(before, state(ltp="23500"))


class TestVolatility:
    def test_a_vix_jump_fires(self):
        before = state(india_vix=11.0)
        events = events_of(
            before, state(india_vix=13.0), TriggerType.VOLATILITY_EXPANSION_CONTRACTION
        )
        assert events and events[0].payload["direction"] == "expansion"

    def test_a_vix_collapse_fires_as_contraction(self):
        before = state(india_vix=18.0)
        events = events_of(
            before, state(india_vix=14.0), TriggerType.VOLATILITY_EXPANSION_CONTRACTION
        )
        assert events and events[0].payload["direction"] == "contraction"

    def test_a_missing_vix_is_not_a_collapse(self):
        """The specific bug this guards: None read as zero reports a total
        volatility collapse every time the feed drops a field."""
        before = state(india_vix=11.34)
        triggers = fired(before, state(india_vix=None, realized_vol=None, atm_iv=None))
        assert TriggerType.VOLATILITY_EXPANSION_CONTRACTION not in triggers
        assert TriggerType.IV_EXPANSION_COLLAPSE not in triggers

    def test_an_iv_collapse_fires(self):
        before = state(atm_iv=14.0)
        events = events_of(before, state(atm_iv=11.0), TriggerType.IV_EXPANSION_COLLAPSE)
        assert events and events[0].payload["direction"] == "collapse"

    def test_a_small_iv_drift_does_not(self):
        assert TriggerType.IV_EXPANSION_COLLAPSE not in fired(
            state(atm_iv=11.0), state(atm_iv=11.2)
        )


class TestOpenInterest:
    def test_a_large_build_fires(self):
        chain = default_chain()
        grown = [
            option(23900, OptionType.CE, oi=140_000, oi_change=50_000)
            if quote.contract.strike == Decimal(23900)
            and quote.contract.option_type is OptionType.CE
            else quote
            for quote in chain
        ]
        events = events_of(state(chain=chain), state(chain=grown), TriggerType.LARGE_OI_ADDITION)
        assert events and events[0].payload["strike"] == 23900.0

    def test_a_large_unwind_fires(self):
        chain = default_chain()
        shrunk = [
            option(23900, OptionType.PE, oi=40_000)
            if quote.contract.strike == Decimal(23900)
            and quote.contract.option_type is OptionType.PE
            else quote
            for quote in chain
        ]
        assert TriggerType.LARGE_OI_UNWINDING in fired(
            state(chain=chain), state(chain=shrunk)
        )

    def test_a_tiny_strike_doubling_is_not_news(self):
        """A strike going from ten lots to twenty has doubled and means
        nothing. A ratio test alone would rank it above a 15% build on the
        ATM strike."""
        small = [option(26000, OptionType.CE, oi=10)]
        doubled = [option(26000, OptionType.CE, oi=20)]
        assert TriggerType.LARGE_OI_ADDITION not in fired(
            state(chain=small), state(chain=doubled)
        )

    def test_a_new_strike_is_not_a_build(self):
        """It has no prior OI to have grown from."""
        assert TriggerType.LARGE_OI_ADDITION not in fired(
            state(chain=[option(23900, OptionType.CE, oi=50_000)]),
            state(
                chain=[
                    option(23900, OptionType.CE, oi=50_000),
                    option(24500, OptionType.CE, oi=80_000),
                ]
            ),
        )

    def test_the_max_oi_strike_moving_fires_migration(self):
        """Where the market thinks the index will settle — a shift in
        consensus rather than a change in magnitude."""
        before = state(
            chain=[
                option(23900, OptionType.CE, oi=90_000),
                option(24000, OptionType.CE, oi=30_000),
            ]
        )
        after = state(
            chain=[
                option(23900, OptionType.CE, oi=90_000),
                option(24000, OptionType.CE, oi=150_000),
            ]
        )
        events = events_of(before, after, TriggerType.OI_MIGRATION)
        assert events and events[0].payload["direction"] == "up"


class TestPremiumAndGamma:
    def test_a_large_atm_premium_move_fires(self):
        before = state(chain=[option(23900, OptionType.CE, bid="100.00", ask="101.00")])
        after = state(chain=[option(23900, OptionType.CE, bid="130.00", ask="131.00")])
        events = events_of(before, after, TriggerType.LARGE_PREMIUM_MOVEMENT)
        assert events and events[0].payload["strike"] == 23900.0

    def test_a_different_atm_strike_is_not_the_same_premium_moving(self):
        """OI migration and price movement cover that case; conflating them
        would double-count one event."""
        before = state(ltp="23900", chain=[option(23900, OptionType.CE, bid="100.00", ask="101.00")])
        after = state(ltp="24000", chain=[option(24000, OptionType.CE, bid="40.00", ask="41.00")])
        assert TriggerType.LARGE_PREMIUM_MOVEMENT not in fired(before, after)

    def test_gamma_concentrating_fires(self):
        spread_out = state(
            chain=[
                option(23800, OptionType.CE, gamma="0.0010"),
                option(23900, OptionType.CE, gamma="0.0010"),
                option(24000, OptionType.CE, gamma="0.0010"),
            ]
        )
        concentrated = state(
            chain=[
                option(23800, OptionType.CE, gamma="0.0002"),
                option(23900, OptionType.CE, gamma="0.0030"),
                option(24000, OptionType.CE, gamma="0.0002"),
            ]
        )
        events = events_of(spread_out, concentrated, TriggerType.GAMMA_CONCENTRATION_CHANGE)
        assert events and events[0].payload["direction"] == "tightening"

    def test_legs_without_greeks_are_skipped(self):
        """Which is the honest outcome for a strike too wide to mark."""
        no_greeks = [
            option(23900, OptionType.CE).model_copy(update={"greeks": None})
        ]
        assert TriggerType.GAMMA_CONCENTRATION_CHANGE not in fired(
            state(chain=no_greeks), state(chain=no_greeks)
        )


class TestLiquidity:
    def test_the_chain_widening_materially_fires(self):
        tight = [
            option(strike, OptionType.CE, bid="100.00", ask="101.00")
            for strike in (23800, 23900, 24000)
        ]
        wide = [
            option(strike, OptionType.CE, bid="90.00", ask="110.00")
            for strike in (23800, 23900, 24000)
        ]
        assert TriggerType.LIQUIDITY_DETERIORATION in fired(
            state(chain=tight), state(chain=wide)
        )

    def test_one_abandoned_wing_does_not_condemn_the_chain(self):
        """Median rather than worst: a worst-case measure fires on a far-wing
        strike every session."""
        tight = [
            option(strike, OptionType.CE, bid="100.00", ask="101.00")
            for strike in (23800, 23900, 24000)
        ]
        one_bad = [
            *tight[:2],
            option(24000, OptionType.CE, bid="1.00", ask="50.00"),
        ]
        assert TriggerType.LIQUIDITY_DETERIORATION not in fired(
            state(chain=tight), state(chain=one_bad)
        )

    def test_an_already_excellent_spread_doubling_is_not_deterioration(self):
        """0.1% to 0.2% has doubled and is still excellent."""
        very_tight = [
            option(strike, OptionType.CE, bid="1000.00", ask="1000.50")
            for strike in (23800, 23900, 24000)
        ]
        slightly_wider = [
            option(strike, OptionType.CE, bid="1000.00", ask="1001.00")
            for strike in (23800, 23900, 24000)
        ]
        assert TriggerType.LIQUIDITY_DETERIORATION not in fired(
            state(chain=very_tight), state(chain=slightly_wider)
        )


class TestConstituents:
    def test_a_heavyweight_move_fires(self):
        weights = {"HDFCBANK": 13.2}
        before = state(constituents=[constituent("HDFCBANK", "1000")], weights=weights)
        after = state(constituents=[constituent("HDFCBANK", "1030")], weights=weights)
        events = events_of(before, after, TriggerType.MAJOR_CONSTITUENT_MOVEMENT)
        assert events and events[0].payload["symbol"] == "HDFCBANK"

    def test_a_small_weight_moving_is_not_news(self):
        """A 3% move in a 0.4%-weight name cannot shift the index, and waking
        a full analysis for it is the noise that makes an event engine worse
        than a timer."""
        weights = {"TINYCO": 0.4}
        before = state(constituents=[constituent("TINYCO", "1000")], weights=weights)
        after = state(constituents=[constituent("TINYCO", "1030")], weights=weights)
        assert TriggerType.MAJOR_CONSTITUENT_MOVEMENT not in fired(before, after)

    def test_breadth_shifting_fires(self):
        before = state(with_analysis=analysis(breadth=-0.4))
        after = state(with_analysis=analysis(breadth=0.4))
        assert TriggerType.BREADTH_CHANGE in fired(before, after)

    def test_no_constituent_provider_means_silence(self):
        """NSE public serves no constituents, so these detectors must be
        inert rather than firing on empty data."""
        triggers = fired(state(), state())
        assert TriggerType.MAJOR_CONSTITUENT_MOVEMENT not in triggers
        assert TriggerType.SECTOR_LEADERSHIP_CHANGE not in triggers

    def test_sector_leadership_changing_fires(self):
        before = state(sector_returns={"Financials": 1.2, "IT": 0.3})
        after = state(sector_returns={"Financials": 0.2, "IT": 1.4})
        events = events_of(before, after, TriggerType.SECTOR_LEADERSHIP_CHANGE)
        assert events and events[0].payload["to_leader"] == "IT"


class TestVolume:
    def test_a_volume_spike_fires(self):
        bars = [bar("23900", offset=n, volume=100_000) for n in range(6)]
        spike = [*bars, bar("23900", offset=6, volume=500_000)]
        assert TriggerType.VOLUME_ANOMALY in fired(
            state(intraday_bars=bars), state(intraday_bars=spike)
        )

    def test_a_feed_reporting_no_volume_produces_no_anomalies(self):
        """NSE's index snapshot carries no volume. A baseline of zeros would
        make every bar an anomaly."""
        bars = [bar("23900", offset=n, volume=0) for n in range(7)]
        assert TriggerType.VOLUME_ANOMALY not in fired(
            state(intraday_bars=bars), state(intraday_bars=bars)
        )


class TestSessionAndTime:
    @pytest.mark.parametrize(
        ("session", "expected"),
        [
            (MarketSessionState.PRE_MARKET, TriggerType.PRE_MARKET),
            (MarketSessionState.OPENING, TriggerType.MARKET_OPEN),
            (MarketSessionState.CLOSING, TriggerType.PRE_CLOSE),
            (MarketSessionState.CLOSED, TriggerType.END_OF_DAY),
        ],
    )
    def test_each_session_transition_has_its_trigger(self, session, expected):
        before = state(session=MarketSessionState.ACTIVE)
        assert expected in fired(before, state(session=session))

    def test_staying_in_a_session_does_not_re_fire(self):
        active = state(session=MarketSessionState.ACTIVE)
        assert TriggerType.MARKET_OPEN not in fired(active, active)

    def test_the_heartbeat_paces_itself_from_snapshot_timestamps(self):
        """Not from a timer, which is what lets a backtest produce the same
        heartbeats as a live session over the same data."""
        engine = DeterministicTriggerEngine(TriggerEngineConfig(heartbeat_seconds=300))
        first = state()
        assert TriggerType.PERIODIC_HEARTBEAT in {
            event.trigger_type for event in engine.detect(None, first)
        }
        soon = state(timestamp=NOW + timedelta(seconds=60))
        assert TriggerType.PERIODIC_HEARTBEAT not in {
            event.trigger_type for event in engine.detect(first, soon)
        }
        later = state(timestamp=NOW + timedelta(seconds=400))
        assert TriggerType.PERIODIC_HEARTBEAT in {
            event.trigger_type for event in engine.detect(soon, later)
        }

    def test_no_heartbeat_outside_the_session(self):
        """It would wake the pipeline all night to re-analyse a closed
        market."""
        engine = DeterministicTriggerEngine()
        closed = state(session=MarketSessionState.CLOSED)
        assert TriggerType.PERIODIC_HEARTBEAT not in {
            event.trigger_type for event in engine.detect(None, closed)
        }

    def test_entering_the_expiry_phase_fires_once(self):
        before = state(days_to_expiry=1.5)
        assert TriggerType.EXPIRY_PHASE in fired(before, state(days_to_expiry=0.8))
        inside = state(days_to_expiry=0.8)
        assert TriggerType.EXPIRY_PHASE not in fired(inside, state(days_to_expiry=0.3))


class TestCalendarTriggers:
    def test_four_triggers_are_unreachable_without_a_calendar(self):
        """Calendar facts, not measurements. No free Indian source for the
        calendar was found, and fabricating the dates would put invented event
        risk into the Risk Engine's blackout logic — refusing to trade on
        quiet days and trading through the ones that matter."""
        engine = DeterministicTriggerEngine()
        assert engine.unreachable_triggers == CALENDAR_ONLY_TRIGGERS
        assert len(CALENDAR_ONLY_TRIGGERS) == 4

    def test_a_calendar_makes_them_reachable(self):
        class Stub(ScheduledEventCalendar):
            def events_between(self, start, end):
                return [
                    ScheduledEvent(name="RBI Monetary Policy", starts_at=end),
                    ScheduledEvent(name="NIFTY Index Rebalance", starts_at=end),
                    ScheduledEvent(name="CPI print", starts_at=end),
                ]

        engine = DeterministicTriggerEngine(calendar=Stub())
        assert engine.unreachable_triggers == frozenset()
        triggers = {event.trigger_type for event in engine.detect(state(), state())}
        assert TriggerType.RBI_EVENT in triggers
        assert TriggerType.INDEX_REBALANCE in triggers
        assert TriggerType.MAJOR_SCHEDULED_ECONOMIC_EVENT in triggers


class TestFailSoft:
    def test_one_broken_detector_does_not_lose_the_others(self):
        """A chain with a malformed leg must not cost you the session
        boundary."""

        def exploding(previous, current, cfg):
            raise ValueError("bad chain leg")

        engine = DeterministicTriggerEngine(
            detectors=[exploding, *DeterministicTriggerEngine()._detectors]
        )
        triggers = {
            event.trigger_type
            for event in engine.detect(
                state(session=MarketSessionState.ACTIVE),
                state(session=MarketSessionState.CLOSING),
            )
        }
        assert TriggerType.PRE_CLOSE in triggers
        assert TriggerType.EXCEPTIONAL_MARKET_EVENT in triggers

    def test_the_failure_is_reported_as_news_not_logged_quietly(self):
        """A detector that cannot read the market is itself worth waking for:
        the state that broke it is exactly the state worth looking at."""

        def exploding(previous, current, cfg):
            raise ValueError("bad chain leg")

        engine = DeterministicTriggerEngine(detectors=[exploding])
        events = engine.detect(state(), state())
        assert len(events) == 1
        assert events[0].trigger_type is TriggerType.EXCEPTIONAL_MARKET_EVENT
        assert "bad chain leg" in events[0].payload["error"]
        assert events[0].payload["detector"] == "exploding"


class TestTriggersCannotTrade:
    def test_the_engine_has_no_route_to_an_order(self):
        """The §4 invariant. A trigger only means "something changed; analyze
        it"."""
        engine = DeterministicTriggerEngine()
        for attribute in ("submit", "place_order", "authorize", "broker", "_broker"):
            assert not hasattr(engine, attribute)

    def test_events_carry_no_price_to_act_on(self):
        """A payload is evidence for the analysis, not an instruction."""
        events = DeterministicTriggerEngine().detect(state(), state(ltp="24010"))
        for event in events:
            assert "quantity" not in event.payload
            assert "side" not in event.payload
            assert "order" not in event.payload


class TestSignificanceFilter:
    def event(self, trigger: TriggerType, score: float, seconds: int = 0) -> Event:
        return Event(
            event_id=f"e{seconds}",
            trigger_type=trigger,
            timestamp=NOW + timedelta(seconds=seconds),
            significance_score=score,
        )

    def test_a_low_score_is_held_back(self):
        f = ThresholdSignificanceFilter(SignificanceFilterConfig(min_score=0.5))
        assert not f.is_significant(
            self.event(TriggerType.SIGNIFICANT_PRICE_MOVEMENT, 0.3)
        )

    def test_a_high_score_passes(self):
        f = ThresholdSignificanceFilter(SignificanceFilterConfig(min_score=0.5))
        assert f.is_significant(self.event(TriggerType.SIGNIFICANT_PRICE_MOVEMENT, 0.8))

    def test_an_unscored_event_is_held_back(self):
        """A detector reporting no magnitude has said nothing about how much
        this matters. Treating that as significant would let it wake the
        pipeline forever."""
        f = ThresholdSignificanceFilter()
        unscored = Event(
            event_id="e", trigger_type=TriggerType.VOLUME_ANOMALY, timestamp=NOW
        )
        assert not f.is_significant(unscored)

    def test_the_cooldown_suppresses_a_repeat(self):
        """Without it the engine is a very expensive timer: a market grinding
        through a level fires on every tick and the pipeline never finishes
        being useful."""
        f = ThresholdSignificanceFilter(
            SignificanceFilterConfig(default_cooldown_seconds=60)
        )
        assert f.is_significant(self.event(TriggerType.SUPPORT_RESISTANCE_TEST, 0.8, 0))
        assert not f.is_significant(
            self.event(TriggerType.SUPPORT_RESISTANCE_TEST, 0.8, 30)
        )
        assert f.is_significant(self.event(TriggerType.SUPPORT_RESISTANCE_TEST, 0.8, 90))

    def test_the_cooldown_is_per_trigger_type(self):
        """A suppressed price move must not suppress an IV collapse arriving
        in the same second."""
        f = ThresholdSignificanceFilter(
            SignificanceFilterConfig(default_cooldown_seconds=60)
        )
        assert f.is_significant(self.event(TriggerType.SIGNIFICANT_PRICE_MOVEMENT, 0.8, 0))
        assert f.is_significant(self.event(TriggerType.IV_EXPANSION_COLLAPSE, 0.8, 1))

    def test_session_boundaries_bypass_both_gates(self):
        """A session boundary changes what every other reading means.
        Suppressing one because something similar fired a minute ago would
        suppress the most important wake-up of the day."""
        f = ThresholdSignificanceFilter(
            SignificanceFilterConfig(min_score=0.9, default_cooldown_seconds=3600)
        )
        assert f.is_significant(self.event(TriggerType.MARKET_OPEN, 0.1, 0))
        assert f.is_significant(self.event(TriggerType.MARKET_OPEN, 0.1, 1))

    def test_an_exceptional_event_is_never_suppressed(self):
        f = ThresholdSignificanceFilter(
            SignificanceFilterConfig(min_score=0.99, default_cooldown_seconds=3600)
        )
        assert f.is_significant(
            self.event(TriggerType.EXCEPTIONAL_MARKET_EVENT, 0.05, 0)
        )

    def test_a_per_trigger_cooldown_override_is_honoured(self):
        f = ThresholdSignificanceFilter(
            SignificanceFilterConfig(
                default_cooldown_seconds=3600,
                cooldown_seconds={"SIGNIFICANT_PRICE_MOVEMENT": 10},
            )
        )
        assert f.is_significant(self.event(TriggerType.SIGNIFICANT_PRICE_MOVEMENT, 0.8, 0))
        assert f.is_significant(self.event(TriggerType.SIGNIFICANT_PRICE_MOVEMENT, 0.8, 20))

    def test_a_held_back_event_explains_itself(self):
        """"Why did the system not react to that" is the question this layer
        gets asked, and a boolean cannot answer it."""
        f = ThresholdSignificanceFilter(SignificanceFilterConfig(min_score=0.5))
        decision = f.evaluate(self.event(TriggerType.VOLUME_ANOMALY, 0.2))
        assert not decision.significant
        assert "below the" in decision.reason

    def test_it_takes_time_from_the_events_not_a_clock(self):
        """So a replay suppresses exactly what the live session suppressed."""
        f = ThresholdSignificanceFilter(
            SignificanceFilterConfig(default_cooldown_seconds=60)
        )
        historic = Event(
            event_id="h1",
            trigger_type=TriggerType.BREAKOUT,
            timestamp=NOW.replace(year=2019),
            significance_score=0.8,
        )
        assert f.is_significant(historic)

    def test_reset_clears_the_cooldown_memory(self):
        """Used at a session boundary, so one day's suppression does not carry
        into the next."""
        f = ThresholdSignificanceFilter(
            SignificanceFilterConfig(default_cooldown_seconds=3600)
        )
        assert f.is_significant(self.event(TriggerType.BREAKOUT, 0.8, 0))
        assert not f.is_significant(self.event(TriggerType.BREAKOUT, 0.8, 10))
        f.reset()
        assert f.is_significant(self.event(TriggerType.BREAKOUT, 0.8, 20))

    def test_filter_returns_only_the_significant_events_in_order(self):
        f = ThresholdSignificanceFilter(SignificanceFilterConfig(min_score=0.5))
        events = [
            self.event(TriggerType.VOLUME_ANOMALY, 0.2, 0),
            self.event(TriggerType.BREAKOUT, 0.9, 1),
            self.event(TriggerType.IV_EXPANSION_COLLAPSE, 0.7, 2),
        ]
        passed = f.filter(events)
        assert [event.trigger_type for event in passed] == [
            TriggerType.BREAKOUT,
            TriggerType.IV_EXPANSION_COLLAPSE,
        ]
