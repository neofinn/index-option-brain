"""The line that carries what a chart cannot compute."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from index_option_brain.brain.pipeline import QuantitativeBrain
from index_option_brain.contracts.market_state import MarketState
from index_option_brain.integrations.tradingview import chartfeed

PINE = (
    Path(__file__).resolve().parents[3]
    / "index_option_brain"
    / "integrations"
    / "tradingview"
    / "pine"
    / "index_brain_mirror.pine"
)

AS_OF = datetime(2026, 9, 6, 4, 15, tzinfo=UTC)


class TestEncoding:
    def test_a_line_starts_with_version_time_and_symbol(self) -> None:
        line = chartfeed.encode({"iv": "11.20"}, as_of=AS_OF, symbol="nifty")
        assert line.startswith("v1|2026-09-06T04:15:00Z|NIFTY|")

    def test_an_unmeasured_field_is_omitted_not_zeroed(self) -> None:
        """A max pain of 0 renders as a line at zero on the chart; an
        omitted one renders as a dash. On a chart the difference between
        absent and zero is the difference between nothing and a lie."""
        line = chartfeed.encode({"mp": None, "iv": "11.20"}, as_of=AS_OF, symbol="NIFTY")
        assert "mp=" not in line
        assert "iv=11.20" in line

    def test_it_stays_one_line_with_no_quotes(self) -> None:
        """It has to survive a paste into a TradingView input box."""
        line = chartfeed.encode({"cw": "24100.00,24200.00"}, as_of=AS_OF, symbol="NIFTY")
        assert "\n" not in line
        assert '"' not in line and "'" not in line

    def test_none_never_becomes_a_number(self) -> None:
        assert chartfeed._number(None) is None
        assert chartfeed._number(float("nan")) is None
        assert chartfeed._number(True) is None
        assert chartfeed._number(Decimal("24025.6512")) == "24025.65"

    def test_an_empty_level_list_is_absent_rather_than_empty(self) -> None:
        assert chartfeed._levels([]) is None
        levels = [Decimal(24100), Decimal(24200)]
        assert chartfeed._levels(levels) == "24100.00,24200.00"


class TestDecoding:
    def test_a_round_trip_preserves_the_fields(self) -> None:
        line = chartfeed.encode(
            {"mp": "24000.00", "vrp": "-1.80"}, as_of=AS_OF, symbol="NIFTY"
        )
        fields = chartfeed.decode(line)
        assert fields["symbol"] == "NIFTY"
        assert fields["mp"] == "24000.00"
        assert fields["vrp"] == "-1.80"

    def test_an_unknown_version_yields_nothing(self) -> None:
        """A malformed paste must not half-parse into plausible numbers."""
        assert chartfeed.decode("v9|2026-09-06T04:15:00Z|NIFTY|mp=24000") == {}

    def test_junk_yields_nothing(self) -> None:
        assert chartfeed.decode("buy nifty now") == {}
        assert chartfeed.decode("") == {}


class TestBuild:
    def test_it_carries_the_chain_readings_the_chart_cannot_see(
        self, uptrend_state: MarketState
    ) -> None:
        result = QuantitativeBrain().run(uptrend_state)
        fields = chartfeed.decode(chartfeed.build(result))
        assert fields["symbol"] == uptrend_state.index_symbol
        # The whole reason the feed exists: none of these is computable in
        # Pine, and all of them are measured here.
        for key in ("mp", "iv", "pcr"):
            assert key in fields

    def test_authorization_is_always_stated(self, uptrend_state: MarketState) -> None:
        """The one field whose absence would read as "probably fine". A
        chart showing levels with no authorization line looks exactly like
        a chart showing an authorized trade."""
        fields = chartfeed.decode(chartfeed.build(QuantitativeBrain().run(uptrend_state)))
        assert fields["auth"] in {"0", "1"}

    def test_it_is_short_enough_to_paste(self, uptrend_state: MarketState) -> None:
        line = chartfeed.build(QuantitativeBrain().run(uptrend_state))
        assert len(line) < 600, f"too long to paste comfortably: {len(line)}"


class TestPineReadsWhatPythonWrites:
    """The drift that actually happens: a key renamed on one side."""

    def test_every_key_the_indicator_reads_is_one_the_encoder_can_emit(
        self, uptrend_state: MarketState
    ) -> None:
        source = PINE.read_text()
        read = set(
            re.findall(r'f_feed(?:_num|_field|_first)\(feed, "([a-z]+)"\)', source)
        )
        assert read, "the indicator reads no feed keys at all"
        written = set(chartfeed.decode(chartfeed.build(QuantitativeBrain().run(uptrend_state))))
        # Fields the engine could not measure this cycle are legitimately
        # absent from one line, so the check is against what `build` knows
        # how to emit rather than against one sample.
        emittable = written | {
            "spot", "regime", "rgc", "dir", "sig", "idx", "sup", "res",
            "adv", "dec", "brd", "cov", "mp", "cw", "pw", "pcr", "xb",
            "iv", "rv", "rvw", "vrp", "ivp", "em", "eam", "strat", "auth",
        }
        assert read <= emittable, f"the indicator reads keys nothing writes: {read - emittable}"

    def test_the_indicator_refuses_a_version_it_does_not_know(self) -> None:
        assert 'feedVersion == "v1"' in PINE.read_text()
        assert chartfeed.FEED_VERSION == "v1"

    def test_the_indicator_withholds_a_stale_feed_rather_than_drawing_it(self) -> None:
        """An old max pain drawn as a fresh line is worse than no line."""
        source = PINE.read_text()
        assert "feedOk ? f_feed_num(feed, \"mp\")" in source
        assert "STALE" in source


class TestCasing:
    def test_direction_is_upper_cased_for_the_indicator(
        self, uptrend_state: MarketState
    ) -> None:
        """Direction's enum values are lower case and the indicator
        compares against "BULLISH"/"BEARISH" to colour the row. A silent
        case mismatch leaves every signal grey."""
        fields = chartfeed.decode(chartfeed.build(QuantitativeBrain().run(uptrend_state)))
        assert fields["dir"] in {"BULLISH", "BEARISH", "NEUTRAL"}

    def test_the_indicator_compares_against_the_same_casing(self) -> None:
        source = PINE.read_text()
        assert 'fDirTxt == "BULLISH"' in source
        assert 'fDirTxt == "BEARISH"' in source
