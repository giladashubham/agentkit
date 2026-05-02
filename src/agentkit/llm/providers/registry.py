from __future__ import annotations

from collections.abc import Callable

from ..model import Model
from .base import Provider

__all__ = [
    "ProviderFactory",
    "register_provider",
    "get_provider",
    "list_provider_apis",
    "clear_providers",
]

ProviderFactory = Callable[[Model], Provider]

_api_provider_factories: dict[str, ProviderFactory] = {}


def register_provider(api: str, factory: ProviderFactory) -> None:
    """Register a provider factory for an API protocol."""
    _api_provider_factories[api] = factory


def get_provider(model: Model) -> Provider:
    """Create a provider for a model using model.api."""
    try:
        factory = _api_provider_factories[model.api]
    except KeyError as exc:
        available = ", ".join(sorted(_api_provider_factories)) or "none"
        raise ValueError(f"Unknown api: {model.api}. Available: {available}") from exc
    return factory(model)


def list_provider_apis() -> list[str]:
    """List registered API names."""
    return sorted(_api_provider_factories)


def clear_providers() -> None:
    """Clear registered providers. Primarily useful for tests."""
    _api_provider_factories.clear()
