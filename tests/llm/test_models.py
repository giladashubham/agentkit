from __future__ import annotations

import pytest

from agentkit.llm import (
    Model,
    ModelCost,
    Usage,
    calculate_cost,
    get_model,
    get_models,
    list_model_providers,
    register_model,
)
from agentkit.llm.models.builtins import register_builtin_models
from agentkit.llm.models.catalog import iter_builtin_models
from agentkit.llm.models.registry import clear_models


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
    assert list_model_providers() == ["custom"]


def test_get_unknown_model_raises() -> None:
    clear_models()

    with pytest.raises(ValueError, match="Unknown model: custom/missing"):
        get_model("custom", "missing")


def test_builtin_catalog_includes_model_metadata() -> None:
    model = get_model("openai", "gpt-4o-mini")

    assert model.name == "GPT-4o mini"
    assert model.context_window == 128_000
    assert model.max_tokens == 16_384
    assert model.input_types == ("text", "image")
    assert model.cost.input > 0
    assert model.cost.output > 0


def test_builtin_catalog_has_unique_provider_model_ids() -> None:
    keys = [(model.provider, model.id) for model in iter_builtin_models()]

    assert len(keys) == len(set(keys))


def test_calculate_cost_updates_usage_cost() -> None:
    model = Model(
        provider="custom",
        api="custom-api",
        id="custom-model",
        cost=ModelCost(input=1.00, output=2.00, cache_read=0.25, cache_write=1.25),
    )
    usage = Usage(input=1_000_000, output=500_000, cache_read=100_000, cache_write=200_000)

    cost = calculate_cost(model, usage)

    assert cost.input == pytest.approx(1.00)
    assert cost.output == pytest.approx(1.00)
    assert cost.cache_read == pytest.approx(0.025)
    assert cost.cache_write == pytest.approx(0.25)
    assert cost.total == pytest.approx(2.275)
    assert usage.cost == cost
