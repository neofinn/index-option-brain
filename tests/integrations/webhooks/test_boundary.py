"""Where the capability to place an order lives, and where it does not.

This file changed when the signal relay was added, and the change is the
point: the gateway *service* still cannot cause an outbound order, but the
process that runs it now can, deliberately, for `strategy` endpoints with
an enabled route. Asserting the old blanket claim would be asserting
something that is no longer true.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from index_option_brain.integrations.webhooks.gateway import create_gateway_app
from tests.integrations.boundary import PACKAGE, reachable_offences

SERVICE_MODULES = (
    "index_option_brain.integrations.webhooks.gateway",
    "index_option_brain.integrations.webhooks.endpoints",
    "index_option_brain.integrations.webhooks.store",
)


def _imports(module: str) -> set[str]:
    relative = Path(*module.split(".")[1:])
    for candidate in (
        PACKAGE / relative.with_suffix(".py"),
        PACKAGE / relative / "__init__.py",
    ):
        if candidate.exists():
            source = candidate.read_text()
            break
    else:  # pragma: no cover - a renamed module should fail loudly
        raise AssertionError(f"no such module: {module}")
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
        elif isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
    return found


class TestTheServiceCannotSend:
    def test_the_service_modules_reach_no_engine_risk_or_broker(self) -> None:
        offences = reachable_offences(*SERVICE_MODULES)
        assert not offences, f"the gateway service can reach order placement: {offences}"

    def test_the_service_modules_do_not_import_the_relay(self) -> None:
        """`gateway.py` takes a generic delivery handler. Keeping the relay
        out of it is what makes the capability auditable: it is wired in
        exactly one file rather than reachable from the request handler."""
        for module in SERVICE_MODULES:
            relay_imports = {
                name for name in _imports(module) if name.startswith("index_option_brain.signals")
            }
            assert not relay_imports, f"{module} imports {relay_imports}"

    def test_the_app_is_built_from_a_registry_and_a_store(self) -> None:
        parameters = inspect.signature(create_gateway_app).parameters
        positional = [
            p.annotation
            for p in parameters.values()
            if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        ]
        assert positional == ["dict[str, Endpoint]", "DeliveryStore"]


class TestTheCapabilityIsPinnedToOneFile:
    def test_only_the_runner_wires_the_relay(self) -> None:
        """Asserted positively. If a second module starts importing the
        relay, the reviewer who has to check "can this place an order"
        gains a second file to read, and this test is the thing that
        notices."""
        package = PACKAGE / "integrations" / "webhooks"
        wiring = {
            path.name
            for path in package.rglob("*.py")
            if "index_option_brain.signals.relay" in path.read_text()
        }
        assert wiring == {"__main__.py"}

    def test_the_runner_does_wire_it(self) -> None:
        assert "index_option_brain.signals.relay" in _imports(
            "index_option_brain.integrations.webhooks.__main__"
        )
