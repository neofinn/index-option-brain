"""The chart and the engine must be computing the same thing.

The Pine indicator is a transliteration of `IndexBrainConfig` and
`brain/indicators.py`. Nothing here executes Pine — TradingView is the only
runtime for that — so these tests guard the failure that actually happens
in practice: someone tunes a default in Python, the chart keeps the old
one, and the two quietly disagree while both look right.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from index_option_brain.brain.config import IndexBrainConfig

PINE = (
    Path(__file__).resolve().parents[3]
    / "index_option_brain"
    / "integrations"
    / "tradingview"
    / "pine"
    / "index_brain_mirror.pine"
)

_INPUT = re.compile(
    r"""input\.(?P<kind>int|float)\(\s*(?P<default>-?[\d.]+)\s*,\s*"(?P<name>[a-z_]+)\"""",
)


@pytest.fixture(scope="module")
def source() -> str:
    return PINE.read_text()


@pytest.fixture(scope="module")
def pine_inputs(source: str) -> dict[str, float]:
    return {
        match.group("name"): float(match.group("default"))
        for match in _INPUT.finditer(source)
    }


class TestConfigParity:
    def test_the_pine_defaults_match_indexbrainconfig(
        self, pine_inputs: dict[str, float]
    ) -> None:
        config = IndexBrainConfig()
        mismatches = [
            f"{name}: pine {value} vs engine {getattr(config, name)}"
            for name, value in pine_inputs.items()
            if hasattr(config, name) and float(getattr(config, name)) != value
        ]
        assert not mismatches, f"the chart disagrees with the engine: {mismatches}"

    @pytest.mark.parametrize(
        "name",
        [
            "ema_fast",
            "ema_slow",
            "slope_period",
            "rsi_period",
            "atr_period",
            "roc_period",
            "swing_lookback",
            "breakout_lookback",
            "ema_separation_scale",
            "slope_atr_scale",
            "roc_scale",
            "breakout_buffer_atr",
            "direction_threshold",
            "min_daily_bars",
        ],
    )
    def test_every_knob_the_brain_reads_is_exposed_on_the_chart(
        self, name: str, pine_inputs: dict[str, float]
    ) -> None:
        """A knob the engine reads but the chart hardcodes is a divergence
        waiting for the first time anyone tunes it."""
        assert name in pine_inputs


class TestItDoesNotUseTheWrongBuiltin:
    """Pine's built-ins are not the engine's functions, and three of them
    are close enough to look right on a chart while disagreeing by several
    points."""

    def test_it_does_not_use_ta_ema(self, source: str) -> None:
        # ta.ema seeds with an SMA of the first `period` bars; the engine
        # seeds with the first observation, and an EMA50 still carries
        # ~9% of its seed after 60 bars.
        assert "ta.ema(" not in source

    def test_it_does_not_use_ta_rsi(self, source: str) -> None:
        # ta.rsi is Wilder's; the engine's is Cutler's.
        assert "ta.rsi(" not in source

    def test_it_does_not_use_ta_atr(self, source: str) -> None:
        # ta.atr applies Wilder's smoothing; the engine takes a simple mean.
        assert "ta.atr(" not in source

    def test_the_breakout_range_excludes_the_live_bar(self, source: str) -> None:
        # A range that includes the bar being tested moves with price and
        # nothing ever breaks out.
        assert "ta.highest(high[1]" in source
        assert "ta.lowest(low[1]" in source


class TestAlertPayloads:
    def test_every_kind_it_sends_is_one_the_receiver_accepts(self, source: str) -> None:
        from index_option_brain.integrations.tradingview.alert import CHART_OBSERVABLE

        sent = set(re.findall(r'f_payload\("([A-Z_]+)"', source))
        assert sent, "the indicator sends no alerts at all"
        assert sent <= {str(trigger) for trigger in CHART_OBSERVABLE}

    def test_the_payload_carries_every_required_field(self, source: str) -> None:
        for field in ("secret", "kind", "ticker", "interval", "price", "bar_time", "fired_at"):
            assert f'"{field}"' in source

    def test_timestamps_are_formatted_as_utc(self, source: str) -> None:
        """alert() does not substitute {{time}}, so the indicator builds
        the timestamps — and the receiver reads them as UTC."""
        assert 'str.format_time(t, "yyyy-MM-dd\'T\'HH:mm:ss\'Z\'", "UTC")' in source

    def test_alerts_fire_on_the_transition_not_the_state(self, source: str) -> None:
        """A state alert repeats every bar while price sits beyond the
        range, and the receiver's dedupe only collapses repeats within one
        bar."""
        assert "dBstate == 1 and dBstate[1] != 1" in source
        assert "dDir != dDir[1]" in source

    def test_it_refuses_to_fire_without_a_real_secret(self, source: str) -> None:
        assert "str.length(tvSecret) >= 16" in source


class TestItSaysWhatItCannotSee:
    def test_the_panel_names_the_blind_spots(self, source: str) -> None:
        """Without this line a green BULLISH on the chart reads as the
        system's verdict rather than as one of its four inputs."""
        assert "NOT VISIBLE HERE" in source
        assert "breadth" in source
        assert "engine only — nothing here trades" in source
