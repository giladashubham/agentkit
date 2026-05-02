from __future__ import annotations

from agentkit.llm import Model, get_env_api_key
from agentkit.llm.providers.builtins import _api_key_for_model


def test_get_env_api_key_returns_known_provider_key() -> None:
    env = {"OPENROUTER_API_KEY": "openrouter-key"}

    assert get_env_api_key("openrouter", env) == "openrouter-key"


def test_get_env_api_key_uses_first_configured_alias() -> None:
    env = {"GEMINI_API_KEY": "gemini-key", "GOOGLE_API_KEY": "google-key"}

    assert get_env_api_key("google", env) == "gemini-key"


def test_get_env_api_key_returns_none_for_missing_or_unknown_provider() -> None:
    assert get_env_api_key("openai", {}) is None
    assert get_env_api_key("custom", {"CUSTOM_API_KEY": "custom-key"}) is None


def test_model_api_key_takes_precedence_over_environment(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    model = Model(
        provider="openai",
        api="openai-completions",
        id="gpt-4o-mini",
        api_key="explicit-key",
    )

    assert _api_key_for_model(model) == "explicit-key"


def test_provider_factories_fall_back_to_environment(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    model = Model(provider="groq", api="openai-completions", id="llama")

    assert _api_key_for_model(model) == "groq-key"
