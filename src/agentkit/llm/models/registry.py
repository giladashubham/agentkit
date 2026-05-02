from __future__ import annotations

from collections import defaultdict

from ..model import Model

__all__ = [
    "register_model",
    "get_model",
    "get_models",
    "list_model_providers",
    "clear_models",
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


def list_model_providers() -> list[str]:
    """List providers with registered models."""
    return sorted(_models)


def clear_models() -> None:
    """Clear registered models. Primarily useful for tests."""
    _models.clear()
