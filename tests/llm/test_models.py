from __future__ import annotations

import pytest

from agentkit.llm import Model, get_model, get_models, get_providers, register_model
from agentkit.llm.registry import clear_models, register_builtin_models


@pytest.fixture(autouse=True)
def restore_builtin_models():
    yield
    clear_models()
    register_builtin_models()


def test_register_and_get_model() -> None:
    clear_models()
    model = Model(provider="custom", api="custom-api", id="custom-model")

    register_model(model)

    assert get_model("custom", "custom-model") == model
    assert get_models("custom") == [model]
    assert get_providers() == ["custom"]


def test_get_unknown_model_raises() -> None:
    clear_models()

    with pytest.raises(ValueError, match="Unknown model: custom/missing"):
        get_model("custom", "missing")
