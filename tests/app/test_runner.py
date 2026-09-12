"""The continuous engine loop.

Driven cycle by cycle against recorded NSE payloads rather than left to run
in the background: a test racing a real loop is a test that fails
intermittently for reasons unrelated to the code.

The behaviour worth protecting is that the loop is *event-driven*. Re-running
a full analysis every cycle on a market that has not moved is exactly what
the §4 trigger contract exists to avoid, and it is an easy thing to
accidentally regress into.
"""

from __future__ import annotations

from index_option_brain.app.runner import MarketPoller, PollerConfig
from index_option_brain.data.http import HttpResponse
from tests.app.test_api import engine_on
from tests.data.conftest import nse_session


def poller(session=None, **config) -> MarketPoller:
    return MarketPoller(
        engine_on(session or nse_session()),
        symbols=("NIFTY",),
        config=PollerConfig(**config),
    )


class TestCycles:
    async def test_a_cycle_reads_state_and_counts_itself(self):
        p = poller()
        state = await p.cycle("NIFTY")
        assert state is not None
        assert p.stats.cycles == 1
        assert p.stats.successful_cycles == 1
        assert p.stats.consecutive_failures == 0

    async def test_the_first_cycle_detects_only_time_triggers(self):
        """Nothing comparative can fire without a previous state."""
        p = poller()
        await p.cycle("NIFTY")
        triggers = {str(event.trigger_type) for event in p.recent_events()}
        assert "SIGNIFICANT_PRICE_MOVEMENT" not in triggers

    async def test_an_unchanged_market_does_not_re_run_the_brain(self):
        """The whole reason for an event engine. The recorded payload is
        identical on every poll, so after the first cycle nothing significant
        happens and the analysis must not run again."""
        p = poller()
        for _ in range(5):
            await p.cycle("NIFTY")
        assert p.stats.cycles == 5
        assert p.stats.analyses_run < p.stats.cycles

    async def test_analysis_can_be_forced_for_debugging(self):
        p = poller(analyse_on_every_cycle=True)
        for _ in range(3):
            await p.cycle("NIFTY")
        assert p.stats.analyses_run == 3

    async def test_the_last_result_is_kept_for_the_console(self):
        p = poller()
        await p.cycle("NIFTY")
        result = p.last_result("NIFTY")
        assert result is not None
        assert result.selected_strategy

    async def test_events_are_returned_newest_first(self):
        """An operator reads the top of the list."""
        p = poller()
        await p.cycle("NIFTY")
        events = p.recent_events()
        if len(events) > 1:
            assert events[0].timestamp >= events[-1].timestamp


class TestFailureHandling:
    async def test_a_blocked_feed_is_counted_not_raised(self):
        """The loop must survive a feed outage: raising would end it, and a
        stopped loop stops accumulating bars."""
        blocked = nse_session(allIndices=HttpResponse(200, "<html>denied</html>"))
        p = poller(blocked)
        assert await p.cycle("NIFTY") is None
        assert p.stats.failed_cycles == 1
        assert p.stats.consecutive_failures == 1
        assert p.stats.last_error

    async def test_health_turns_false_after_three_consecutive_failures(self):
        """A process alive but blind looks identical to a healthy one from
        /health, which is why /ready exists."""
        blocked = nse_session(allIndices=HttpResponse(200, "<html>denied</html>"))
        p = poller(blocked)
        p.stats.started_at = __import__("datetime").datetime.now(
            __import__("datetime").UTC
        )
        assert p.stats.healthy
        for _ in range(3):
            await p.cycle("NIFTY")
        assert not p.stats.healthy

    async def test_a_recovery_clears_the_failure_streak(self):
        p = poller()
        p.stats.consecutive_failures = 5
        p.stats.last_error = "old failure"
        await p.cycle("NIFTY")
        assert p.stats.consecutive_failures == 0
        assert p.stats.last_error is None

    async def test_an_unknown_symbol_does_not_kill_the_loop(self):
        p = MarketPoller(engine_on(nse_session()), symbols=("FINNIFTY",))
        session = await p._cycle_all()
        assert session is not None
        assert p.stats.failed_cycles >= 1


class TestSnapshot:
    async def test_it_reports_counted_figures(self):
        """Counted, not estimated. The console shows these so an operator can
        tell "running and quiet" from "running and broken"."""
        p = poller()
        await p.cycle("NIFTY")
        snap = p.snapshot()
        assert snap["cycles"] == 1
        assert snap["successful_cycles"] == 1
        assert snap["failed_cycles"] == 0
        assert snap["symbols"] == ["NIFTY"]

    async def test_it_reports_which_triggers_are_unreachable(self):
        """The four calendar-only triggers, surfaced rather than left as
        silence."""
        p = poller()
        await p.cycle("NIFTY")
        assert set(p.snapshot()["unreachable_triggers"]) == {
            "BUDGET_EVENT_RISK",
            "INDEX_REBALANCE",
            "MAJOR_SCHEDULED_ECONOMIC_EVENT",
            "RBI_EVENT",
        }

    def test_an_unstarted_poller_reports_no_uptime(self):
        """Not a zero. A loop that has never run has no uptime to report."""
        p = poller()
        assert p.snapshot()["uptime_seconds"] is None
        assert not p.snapshot()["running"]


class TestItCannotTrade:
    def test_the_loop_has_no_broker_and_no_order_path(self):
        """A loop authorizing sizes against an invented balance would be the
        most dangerous thing in the repository."""
        p = poller()
        for attribute in ("broker", "_broker", "submit", "place_order", "order_manager"):
            assert not hasattr(p, attribute)

    async def test_analysis_from_the_loop_is_never_authorized(self):
        """No account and no portfolio reach the brain, so the Risk Engine
        cannot run and nothing can present itself as authorized."""
        p = poller(analyse_on_every_cycle=True)
        await p.cycle("NIFTY")
        result = p.last_result("NIFTY")
        assert result is not None
        assert result.is_authorized is False
        assert result.risk_decision is None


class TestLifecycle:
    async def test_start_and_stop_are_idempotent(self):
        p = poller(active_interval_seconds=3600, closed_interval_seconds=3600)
        await p.start()
        assert p.running
        await p.start()  # second start must not spawn a second loop
        await p.stop()
        assert not p.running
        await p.stop()  # stopping twice is harmless

    async def test_stopping_ends_the_task_promptly(self):
        p = poller(active_interval_seconds=3600, closed_interval_seconds=3600)
        await p.start()
        await p.stop()
        assert not p.running
