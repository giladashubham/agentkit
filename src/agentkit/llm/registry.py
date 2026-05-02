from __future__ import annotations

from collections import defaultdict

from .model import Model

__all__ = [
    "register_model",
    "get_model",
    "get_models",
    "get_providers",
    "clear_models",
    "register_builtin_models",
]

_models: dict[str, dict[str, Model]] = defaultdict(dict)


def register_model(model: Model) -> None:
    """Register a model for lookup by provider and id."""
    _models[model.provider][model.id] = model


def get_model(provider: str, model_id: str) -> Model:
    """Get a registered model."""
    try:
        return _models[provider][model_id]
    except KeyError as exc:
        raise ValueError(f"Unknown model: {provider}/{model_id}") from exc


def get_models(provider: str) -> list[Model]:
    """List registered models for a provider."""
    return list(_models.get(provider, {}).values())


def get_providers() -> list[str]:
    """List providers with registered models."""
    return sorted(_models)


def clear_models() -> None:
    """Clear model registry. Primarily useful for tests."""
    _models.clear()


def register_builtin_models() -> None:
    """Register a small built-in model set.

    This is intentionally tiny, not generated. Users can register their own models.
    """
    register_model(Model(provider="openai", api="openai-completions", id="gpt-4o-mini"))
    register_model(Model(provider="openai", api="openai-completions", id="gpt-4o"))
    register_model(
        Model(provider="anthropic", api="anthropic-messages", id="claude-sonnet-4-20250514")
    )


register_builtin_models()
