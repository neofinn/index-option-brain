"""A transitive import walk over this package's own modules.

Used by both integration packages to prove the same property: a module that
receives untrusted input from the public internet cannot reach the code
that places an order. The claim is deliberately narrow — it walks
`index_option_brain.*` imports only, so it proves nothing about the
standard library or third-party packages, and it is a structural check
rather than a runtime one. That is still the check worth having: the way
this boundary would actually erode is somebody adding a convenient import.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2] / "index_option_brain"

FORBIDDEN_SEGMENTS = ("execution", "risk", "broker", "order")


def _module_path(name: str) -> Path | None:
    relative = Path(*name.split(".")[1:])
    for candidate in (
        PACKAGE / relative.with_suffix(".py"),
        PACKAGE / relative / "__init__.py",
    ):
        if candidate.exists():
            return candidate
    return None


def reachable_offences(*roots: str) -> list[str]:
    """Every `index_option_brain` import, transitively from `roots`, whose
    dotted path contains a forbidden segment."""
    seen: set[str] = set()
    frontier = list(roots)
    offences: list[str] = []

    while frontier:
        name = frontier.pop()
        if name in seen:
            continue
        seen.add(name)
        path = _module_path(name)
        if path is None:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            targets: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module:
                targets.append(node.module)
            elif isinstance(node, ast.Import):
                targets.extend(alias.name for alias in node.names)
            for target in targets:
                if not target.startswith("index_option_brain"):
                    continue
                if any(part in FORBIDDEN_SEGMENTS for part in target.split(".")):
                    offences.append(f"{name} imports {target}")
                frontier.append(target)
    return offences
