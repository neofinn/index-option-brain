
from index_option_brain.config.settings import RunMode, Settings


def test_llm_enabled_defaults_to_false(monkeypatch):
    """Spec §23/§35 non-negotiable: the system must remain fully functional
    with LLM_ENABLED=false, and that must be the default, not opt-in."""
    monkeypatch.delenv("LLM_ENABLED", raising=False)
    settings = Settings(_env_file=None)
    assert settings.llm_enabled is False


def test_llm_enabled_can_be_turned_on_explicitly():
    settings = Settings(_env_file=None, LLM_ENABLED=True)
    assert settings.llm_enabled is True


def test_run_mode_defaults_to_paper():
    settings = Settings(_env_file=None)
    assert settings.run_mode is RunMode.PAPER
