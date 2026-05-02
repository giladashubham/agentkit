from __future__ import annotations

import pytest

from agentkit.llm import OPENAI_COMPATIBLE_BASE_URLS, Model, openai_compatible_model
from agentkit.llm.models.presets import with_openai_compatible_base_url


def test_openai_compatible_model_uses_preset_base_url() -> None:
    model = openai_compatible_model("groq", "llama-3.3-70b-versatile")

    assert model.provider == "groq"
    assert model.api == "openai-completions"
    assert model.base_url == OPENAI_COMPATIBLE_BASE_URLS["groq"]


def test_openai_compatible_model_requires_known_provider_or_base_url() -> None:
    with pytest.raises(ValueError, match="Unknown OpenAI-compatible provider"):
        openai_compatible_model("custom", "model")

    model = openai_compatible_model("custom", "model", base_url="https://example.com/v1")
    assert model.base_url == "https://example.com/v1"


def test_with_openai_compatible_base_url_preserves_explicit_base_url() -> None:
    model = Model(provider="groq", api="openai-completions", id="x", base_url="https://custom")

    assert with_openai_compatible_base_url(model).base_url == "https://custom"
