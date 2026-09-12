"""The wall between the investigation layer and every decision.

Spec §23 says an agent may not override risk, the execution gate, position
limits or maximum loss. A docstring saying so is worth nothing — the
guarantee has to be that **no code path exists**, and that is what these
tests assert. Adding one fails the suite rather than a review.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[2] / "index_option_brain"

# Everything that decides, sizes, or sends. If any of these ever imports the
# agent package, an assessment has a route to a trade.
DECISION_PACKAGES = ("brain", "risk", "execution", "state", "events")


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def decision_modules() -> list[Path]:
    return [
        path
        for package in DECISION_PACKAGES
        for path in (PACKAGE / package).rglob("*.py")
    ]


class TestNoRouteFromAgentToDecision:
    def test_there_are_decision_modules_to_check(self):
        """Guards the guard: a glob that silently matched nothing would make
        every test below pass for the wrong reason."""
        assert len(decision_modules()) > 15

    @pytest.mark.parametrize("path", decision_modules(), ids=lambda p: p.name)
    def test_no_decision_module_imports_the_agent(self, path: Path):
        offending = {
            name
            for name in imported_modules(path)
            if name.startswith("index_option_brain.agent")
        }
        assert not offending, (
            f"{path.relative_to(PACKAGE)} imports {sorted(offending)}. An "
            "assessment must have no route to a decision — see spec §23."
        )

    def test_the_agent_package_cannot_reach_risk_or_execution_either(self):
        """The wall is two-way. An agent that could import the Risk Engine
        could call it, and a tool surface that returns a RiskDecision is one
        refactor from producing one."""
        forbidden = ("index_option_brain.risk", "index_option_brain.execution")
        for path in (PACKAGE / "agent").rglob("*.py"):
            offending = {
                name
                for name in imported_modules(path)
                if name.startswith(forbidden)
            }
            assert not offending, f"{path.name} imports {sorted(offending)}"


class TestAssessmentsCarryNoNumbers:
    """An assessment can be read, shown and stored. It cannot be multiplied
    by anything, which is what stops it becoming a score."""

    def test_every_field_is_text(self):
        from index_option_brain.agent.intelligence_provider import AgentAssessment

        for name, field in AgentAssessment.model_fields.items():
            annotation = str(field.annotation)
            assert "float" not in annotation and "int" not in annotation, (
                f"AgentAssessment.{name} is numeric. A number here invites "
                "being multiplied into a decision."
            )

    def test_there_is_no_recommendation_field(self):
        """A recommendation is one step from a decision, and a field named
        for it is an invitation to wire it in."""
        from index_option_brain.agent.intelligence_provider import AgentAssessment

        assert "recommendation" not in AgentAssessment.model_fields
        assert "confidence" not in AgentAssessment.model_fields

    def test_an_assessment_is_immutable(self):
        from pydantic import ValidationError

        from index_option_brain.agent.intelligence_provider import AgentAssessment

        assessment = AgentAssessment(summary="x")
        with pytest.raises(ValidationError):
            assessment.summary = "y"  # type: ignore[misc]


class TestTheDefaultProviderIsReal:
    async def test_it_is_available_with_no_llm_configured(self):
        """Not a stub that raises and not a placeholder returning something
        plausible: the system running normally with this installed is the
        demonstration that no AIProvider is required (spec §23, §35)."""
        from index_option_brain.agent.intelligence_provider import DeterministicProvider

        assessment = await DeterministicProvider().analyze(object())
        assert assessment.provider == "deterministic"
        assert assessment.is_empty
        assert "deterministic" in assessment.summary.lower()

    def test_llm_is_off_by_default(self):
        from index_option_brain.config.settings import Settings

        assert Settings().llm_enabled is False

    async def test_the_whole_pipeline_runs_with_it(self, uptrend_state):
        """The actual §35 requirement: the engine produces a decision with no
        intelligence provider anywhere in the call path."""
        from index_option_brain.brain.pipeline import QuantitativeBrain

        result = QuantitativeBrain().run(uptrend_state)
        assert result.selected_strategy is not None
        assert result.signal is not None
