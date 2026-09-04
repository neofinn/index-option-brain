"""The deploy control loop, and mostly the part of it that refuses.

A control loop driven by a file in a git repository has one obvious failure
mode: whoever can write to the repository can make the machine do anything.
On a box that holds broker credentials, "anything" has to exclude trading —
and these tests are what keeps that true.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from index_option_brain.deploy.reconcile import (
    ALLOWED_KEYS,
    FORBIDDEN_KEYS,
    DesiredStateError,
    load_desired_state,
    plan_from,
    validate,
)


def state(**settings: object) -> dict[str, object]:
    return {"revision": "r1", "settings": settings}


class TestTradingCannotBeEnabledFromGit:
    """The property the whole design rests on."""

    @pytest.mark.parametrize(
        "key",
        ["run_mode", "dry_run", "kill_switch_enabled", "live_trading", "broker_enabled"],
    )
    def test_switches_that_turn_analysis_into_orders_are_refused(self, key: str) -> None:
        with pytest.raises(DesiredStateError, match="forbidden"):
            validate(state(**{key: "anything"}))

    @pytest.mark.parametrize(
        "key",
        [
            "dhan_access_token",
            "delta_api_key",
            "delta_api_secret",
            "anthropic_api_key",
            "database_url",
            "password",
            "secret",
        ],
    )
    def test_credentials_are_refused(self, key: str) -> None:
        with pytest.raises(DesiredStateError, match="forbidden"):
            validate(state(**{key: "x"}))

    def test_case_does_not_evade_the_check(self) -> None:
        """RUN_MODE and run_mode are the same switch."""
        for spelling in ("RUN_MODE", "Run_Mode", "DRY_RUN"):
            with pytest.raises(DesiredStateError, match="forbidden"):
                validate(state(**{spelling: "live"}))

    def test_the_whole_document_is_rejected_not_just_the_bad_key(self) -> None:
        """Filtering would let someone probe the boundary one key at a time
        while their other changes kept landing."""
        with pytest.raises(DesiredStateError) as caught:
            validate(state(branch="main", capture_enabled=True, dry_run=False))
        assert "rejected in full" in str(caught.value)

    def test_the_refusal_says_what_would_be_required_instead(self) -> None:
        with pytest.raises(DesiredStateError, match="human editing .env"):
            validate(state(dry_run=False))

    def test_forbidden_and_allowed_do_not_overlap(self) -> None:
        """A key in both lists would make the guard depend on check order."""
        assert not (FORBIDDEN_KEYS & ALLOWED_KEYS)


class TestUnknownKeys:
    def test_an_unrecognised_setting_is_refused_not_ignored(self) -> None:
        """So a typo does not look like it was applied."""
        with pytest.raises(DesiredStateError, match="unrecognised"):
            validate(state(captrue_enabled=True))

    def test_the_error_lists_what_is_allowed(self) -> None:
        with pytest.raises(DesiredStateError, match="capture_enabled"):
            validate(state(nonsense=1))


class TestRevision:
    def test_a_state_without_a_revision_is_refused(self) -> None:
        """A status report that cannot name the intent it applied is not
        answerable."""
        with pytest.raises(DesiredStateError, match="revision"):
            validate({"settings": {"branch": "main"}})

    def test_a_restart_happens_once_per_revision(self) -> None:
        """`restart: true` left in the file would otherwise restart the
        service on every poll, which looks like a crash loop from outside."""
        desired = validate(state(branch="main", restart=True))
        first = plan_from(desired, current_branch="main", applied_revision=None)
        again = plan_from(desired, current_branch="main", applied_revision="r1")

        assert first.restart is True
        assert again.restart is False


class TestPlanning:
    def test_a_branch_change_is_planned(self) -> None:
        plan = plan_from(validate(state(branch="release")), current_branch="main")
        assert plan.target_branch == "release"
        assert any("release" in r for r in plan.reasons)

    def test_settings_already_in_place_produce_no_work(self) -> None:
        plan = plan_from(
            validate(state(branch="main", capture_enabled=True)),
            current_branch="main",
            current_env={"CAPTURE_ENABLED": "true"},
        )
        assert plan.is_noop

    def test_a_changed_setting_forces_a_restart(self) -> None:
        plan = plan_from(
            validate(state(capture_chain_seconds=60)),
            current_branch="main",
            current_env={"CAPTURE_CHAIN_SECONDS": "300"},
        )
        assert plan.env_updates == {"CAPTURE_CHAIN_SECONDS": "60"}
        assert plan.restart is True

    def test_booleans_render_the_way_the_app_reads_them(self) -> None:
        plan = plan_from(
            validate(state(capture_enabled=False)), current_branch="main"
        )
        assert plan.env_updates["CAPTURE_ENABLED"] == "false"


class TestLoading:
    def test_the_shipped_desired_state_is_valid(self) -> None:
        """The file in the repo must itself pass the guard."""
        desired = load_desired_state(Path("deploy/desired_state.json"))
        assert desired.revision
        assert desired.branch

    def test_a_missing_file_is_an_error_not_an_empty_state(self, tmp_path: Path) -> None:
        with pytest.raises(DesiredStateError, match="No desired state"):
            load_desired_state(tmp_path / "nope.json")

    def test_malformed_json_names_the_file(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        with pytest.raises(DesiredStateError, match="not valid JSON"):
            load_desired_state(bad)


class TestStatusReport:
    def test_it_carries_the_revision_it_applied(self) -> None:
        """A box reporting healthy without saying which intent it is healthy
        under is not answerable."""
        from datetime import UTC, datetime

        from index_option_brain.deploy.reconcile import StatusReport

        report = StatusReport(
            hostname="box",
            reported_at=datetime(2026, 9, 4, tzinfo=UTC),
            applied_revision="r1",
            commit="abc1234",
            branch="main",
            healthy=True,
            service_active=True,
        )
        payload = json.loads(report.to_json())
        assert payload["applied_revision"] == "r1"
        assert payload["healthy"] is True
        assert payload["reported_at"].startswith("2026-09-04")

    def test_it_carries_no_secrets(self) -> None:
        """It is pushed to a branch anyone with repo access can read."""
        from datetime import UTC, datetime

        from index_option_brain.deploy.reconcile import StatusReport

        payload = json.loads(
            StatusReport(
                hostname="box",
                reported_at=datetime.now(UTC),
                applied_revision="r1",
                commit="abc",
                branch="main",
                healthy=False,
                service_active=False,
                errors=["not ready"],
            ).to_json()
        )
        flat = json.dumps(payload).lower()
        for forbidden in ("token", "secret", "password", "api_key"):
            assert forbidden not in flat
