"""Command line the deploy agent shells out to.

Kept in Python rather than in the agent script because the interesting
logic — what a desired state may and may not ask for — deserves tests, and
shell is where untested logic goes to hide.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

from index_option_brain.deploy.reconcile import (
    DesiredStateError,
    StatusReport,
    load_desired_state,
    plan_from,
)


def _plan(args: argparse.Namespace) -> int:
    """Print the plan as shell-eval-able assignments, or fail loudly."""
    try:
        desired = load_desired_state(args.file)
    except DesiredStateError as exc:
        print(f"REJECTED={json.dumps(str(exc))}", file=sys.stdout)
        print(str(exc), file=sys.stderr)
        return 2

    plan = plan_from(
        desired,
        current_branch=args.current_branch,
        current_env=dict(os.environ),
        applied_revision=args.applied_revision or None,
    )
    print(f"REVISION={json.dumps(desired.revision)}")
    print(f"TARGET_BRANCH={json.dumps(plan.target_branch)}")
    print(f"RESTART={'1' if plan.restart else '0'}")
    print(f"ENV_UPDATES={json.dumps(json.dumps(plan.env_updates))}")
    print(f"REASONS={json.dumps('; '.join(plan.reasons))}")
    return 0


def _status(args: argparse.Namespace) -> int:
    detail: dict[str, object] = {}
    if args.detail:
        try:
            detail = json.loads(args.detail)
        except json.JSONDecodeError:
            detail = {"raw": args.detail}

    report = StatusReport(
        hostname=platform.node(),
        reported_at=datetime.now(UTC),
        applied_revision=args.revision,
        commit=args.commit,
        branch=args.branch,
        healthy=args.healthy,
        service_active=args.service_active,
        detail=detail,
        errors=[e for e in (args.error or []) if e],
    )
    Path(args.out).write_text(report.to_json() + "\n")
    print(report.to_json())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="index-brain-deploy")
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="Validate a desired state and print a plan")
    plan.add_argument("--file", required=True)
    plan.add_argument("--current-branch", required=True)
    plan.add_argument("--applied-revision", default="")
    plan.set_defaults(func=_plan)

    status = sub.add_parser("status", help="Write a status report")
    status.add_argument("--out", required=True)
    status.add_argument("--revision", default="")
    status.add_argument("--commit", default="")
    status.add_argument("--branch", default="")
    status.add_argument("--healthy", action="store_true")
    status.add_argument("--service-active", action="store_true")
    status.add_argument("--detail", default="")
    status.add_argument("--error", action="append")
    status.set_defaults(func=_status)

    args = parser.parse_args(argv)
    result = args.func(args)
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
