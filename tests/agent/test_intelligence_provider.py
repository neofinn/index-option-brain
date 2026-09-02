from index_option_brain.agent.intelligence_provider import (
    AgentAssessment,
    DeterministicProvider,
    IntelligenceProvider,
)


async def test_deterministic_provider_works_with_llm_enabled_false():
    """The trading engine must never require an AIProvider to exist
    (spec §23) — DeterministicProvider is the always-on default."""
    provider: IntelligenceProvider = DeterministicProvider()
    assessment = await provider.analyze(context=None)
    assert isinstance(assessment, AgentAssessment)
    assert "deterministic" in assessment.summary.lower()


def test_agent_assessment_has_no_execution_authority_fields():
    """Structural guard: an AgentAssessment can only ever carry
    investigation/reasoning output. If someone later adds an `approved`,
    `order`, or `override_risk`-shaped field here, that is the LLM gaining
    authority the spec explicitly forbids (spec §23)."""
    forbidden_substrings = ("approve", "order", "override", "execute", "quantity")
    field_names = set(AgentAssessment.model_fields)
    for field in field_names:
        for forbidden in forbidden_substrings:
            assert forbidden not in field.lower(), field
