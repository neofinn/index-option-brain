"""Turn a declarative desired state into a plan the box may safely apply.

The deploy agent is a control loop over git: a `desired_state.json` in the
repository says what the machine should be doing, the agent reconciles the
machine to it, and the agent pushes a status report back to a branch. That
gives whoever holds push access remote control of the box without the box
accepting a single inbound connection.

Which is exactly why the interesting code here is the part that says no.

What this may never do
----------------------
A control loop driven by a file in a git repository has one obvious failure
mode: whoever can write to the repository can make the machine do anything.
On a box that will hold broker credentials, "anything" has to exclude
trading. So:

* `FORBIDDEN_KEYS` can never be set by a desired state. They are the
  switches that turn analysis into orders — run mode, kill switch, the
  broker's dry-run flag, and every credential name. A desired state naming
  one is **rejected in full**, not filtered: a caller who asked for
  something forbidden has demonstrated they believe they can, and applying
  the rest of their request silently teaches them they nearly can.
* Enabling live trading therefore requires a human editing `.env` on the
  machine. It cannot be done by pushing a commit, which means it cannot be
  done by anyone who compromises the repository, and cannot be done by me.

This mirrors the rule the brains already follow — no AI may override risk,
execution or position limits — applied one layer down, to the machine
itself rather than to a decision it makes.

Everything else is fair game: which branch to track, how often to poll,
whether capture runs, log levels, which symbols to watch. Those change what
the system *observes*, never what it *does with money*.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: Settings a pushed desired state may never touch. Matched
#: case-insensitively against both the literal key and any env var it would
#: set, because `run_mode` and `RUN_MODE` are the same switch.
FORBIDDEN_KEYS: frozenset[str] = frozenset(
    {
        # Turning analysis into orders
        "run_mode",
        "dry_run",
        "kill_switch_enabled",
        "live_trading",
        "broker_enabled",
        # Credentials of any kind
        "dhan_client_id",
        "dhan_access_token",
        "delta_api_key",
        "delta_api_secret",
        "anthropic_api_key",
        "database_url",
        "api_key",
        "api_secret",
        "access_token",
        "password",
        "secret",
    }
)

#: Settings a desired state may set. An allowlist rather than a denylist,
#: because a denylist silently permits whatever nobody thought of, and the
#: cost of forgetting an entry here is a config that needs a human once.
ALLOWED_KEYS: frozenset[str] = frozenset(
    {
        "branch",
        "poll_seconds",
        "capture_enabled",
        "capture_chain_seconds",
        "symbols",
        "log_level",
        "daily_history_bars",
        "restart",
    }
)


class DesiredStateError(ValueError):
    """A desired state that must not be applied, with the reason."""


@dataclass(frozen=True)
class DesiredState:
    """What the machine has been asked to be.

    `revision` exists so the status report can name what it applied. A box
    reporting "healthy" without saying which intent it is healthy *under* is
    not answerable.
    """

    revision: str
    settings: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    @property
    def branch(self) -> str | None:
        value = self.settings.get("branch")
        return str(value) if value else None

    @property
    def wants_restart(self) -> bool:
        """Whether this revision asks for a restart even with no code change.

        Keyed on the revision rather than a boolean so a restart happens
        once. A plain `true` left in the file would restart the service on
        every poll, which looks like a crash loop from the outside.
        """
        return bool(self.settings.get("restart"))


def _offending(keys: list[str]) -> list[str]:
    lowered = {key.lower() for key in keys}
    return sorted(lowered & FORBIDDEN_KEYS)


def validate(payload: dict[str, Any]) -> DesiredState:
    """Parse and check a desired state, or raise.

    Rejects the whole document on any forbidden key rather than dropping it.
    Partial application would leave the machine in a state nobody described,
    and would let someone probe the boundary one key at a time while their
    other changes kept landing.
    """
    if not isinstance(payload, dict):
        raise DesiredStateError("Desired state must be a JSON object")

    revision = str(payload.get("revision") or "").strip()
    if not revision:
        raise DesiredStateError(
            "Desired state has no revision; a status report that cannot name "
            "the intent it applied is not answerable"
        )

    settings = payload.get("settings")
    if settings is None:
        settings = {}
    if not isinstance(settings, dict):
        raise DesiredStateError("'settings' must be an object")

    offending = _offending(list(settings))
    if offending:
        raise DesiredStateError(
            f"Desired state names forbidden setting(s): {', '.join(offending)}. "
            "Enabling live trading or supplying credentials requires a human "
            "editing .env on the machine — it cannot be done by pushing a "
            "commit, and this document is rejected in full rather than "
            "filtered."
        )

    unknown = sorted({k.lower() for k in settings} - ALLOWED_KEYS)
    if unknown:
        raise DesiredStateError(
            f"Desired state names unrecognised setting(s): {', '.join(unknown)}. "
            f"Allowed: {', '.join(sorted(ALLOWED_KEYS))}. Refusing rather than "
            "ignoring them, so a typo does not look like it was applied."
        )

    return DesiredState(
        revision=revision,
        settings={str(k).lower(): v for k, v in settings.items()},
        note=str(payload.get("note") or ""),
    )


def load_desired_state(path: Path | str) -> DesiredState:
    file = Path(path)
    if not file.exists():
        raise DesiredStateError(f"No desired state at {file}")
    try:
        payload = json.loads(file.read_text())
    except json.JSONDecodeError as exc:
        raise DesiredStateError(f"Desired state at {file} is not valid JSON: {exc}") from exc
    return validate(payload)


@dataclass(frozen=True)
class ReconcilePlan:
    """What the agent should do, given a desired state and where it is now."""

    target_branch: str
    restart: bool
    env_updates: dict[str, str] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    @property
    def is_noop(self) -> bool:
        return not self.restart and not self.env_updates


#: Settings that map onto process environment variables the app reads.
_ENV_MAP = {
    "capture_enabled": "CAPTURE_ENABLED",
    "capture_chain_seconds": "CAPTURE_CHAIN_SECONDS",
    "log_level": "LOG_LEVEL",
    "daily_history_bars": "DAILY_HISTORY_BARS",
    "poll_seconds": "POLL_SECONDS",
}


def plan_from(
    desired: DesiredState,
    *,
    current_branch: str,
    current_env: dict[str, str] | None = None,
    applied_revision: str | None = None,
) -> ReconcilePlan:
    """Diff the desired state against the machine.

    `applied_revision` is the revision the box last acted on. A restart is
    requested per revision rather than per poll, so leaving `restart: true`
    in the file does not restart the service every ten minutes — which from
    the outside is indistinguishable from a crash loop.
    """
    current_env = current_env or {}
    reasons: list[str] = []

    target_branch = desired.branch or current_branch
    if target_branch != current_branch:
        reasons.append(f"branch {current_branch} -> {target_branch}")

    env_updates: dict[str, str] = {}
    for key, env_name in _ENV_MAP.items():
        if key not in desired.settings:
            continue
        value = desired.settings[key]
        rendered = (
            str(value).lower() if isinstance(value, bool) else str(value)
        )
        if current_env.get(env_name) != rendered:
            env_updates[env_name] = rendered
            reasons.append(f"{env_name}={rendered}")

    restart = bool(env_updates)
    if desired.wants_restart and applied_revision != desired.revision:
        restart = True
        reasons.append(f"restart requested by revision {desired.revision}")

    return ReconcilePlan(
        target_branch=target_branch,
        restart=restart,
        env_updates=env_updates,
        reasons=reasons,
    )


@dataclass(frozen=True)
class StatusReport:
    """What the machine reports back, and the only way anyone remote sees it.

    Pushed to a status branch rather than served, so the box still accepts no
    inbound connections. Everything here is operational — no market data, no
    positions, and deliberately nothing from `.env`.
    """

    hostname: str
    reported_at: datetime
    applied_revision: str
    commit: str
    branch: str
    healthy: bool
    service_active: bool
    detail: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "hostname": self.hostname,
                "reported_at": self.reported_at.astimezone(UTC).isoformat(),
                "applied_revision": self.applied_revision,
                "commit": self.commit,
                "branch": self.branch,
                "healthy": self.healthy,
                "service_active": self.service_active,
                "detail": self.detail,
                "errors": self.errors,
            },
            indent=1,
            sort_keys=True,
        )
