"""What the gateway is structurally unable to do."""

from __future__ import annotations

import inspect

from index_option_brain.integrations.webhooks.gateway import create_gateway_app
from tests.integrations.boundary import reachable_offences


class TestItCannotTrade:
    def test_no_module_here_reaches_execution_risk_or_a_broker(self) -> None:
        """A delivery cannot become an order. Turning one into a decision
        is the engine's job, done by polling this API like any other
        consumer."""
        offences = reachable_offences(
            "index_option_brain.integrations.webhooks",
            "index_option_brain.integrations.webhooks.gateway",
            "index_option_brain.integrations.webhooks.__main__",
        )
        assert not offences, f"the gateway can reach order placement: {offences}"

    def test_the_app_is_built_from_a_registry_and_a_store(self) -> None:
        """Nothing it is handed could place an order, so no argument can
        smuggle one in."""
        parameters = inspect.signature(create_gateway_app).parameters
        positional = [
            p.annotation
            for p in parameters.values()
            if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        ]
        assert positional == ["dict[str, Endpoint]", "DeliveryStore"]
