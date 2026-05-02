from __future__ import annotations

from collections.abc import Callable

from ..model import Model
from .base import Provider

ProviderFactory = Callable[[Model], Provider]

_api_provider_factories: dict[str, ProviderFactory] = {}


def register_provider(api: str, factory: ProviderFactory) -> None:
    """Register a provider factory for an API.

    This mirrors Pi's API-provider registry: model.api selects the implementation.
    """
    _api_provider_factories[api] = factory


def get_provider(model: Model) -> Provider:
    """Create a provider for a model using the API provider registry."""
    try:
        factory = _api_provider_factories[model.api]
    except KeyError as exc:
        available = ", ".join(sorted(_api_provider_factories)) or "none"
        raise ValueError(f"Unknown api: {model.api}. Available: {available}") from exc
    return factory(model)


def list_providers() -> list[str]:
    """List registered API provider names."""
    return sorted(_api_provider_factories)


def clear_providers() -> None:
    """Clear registered providers. Primarily useful for tests."""
    _api_provider_factories.clear()


def register_builtin_providers() -> None:
    """Register built-in provider factories lazily."""

    def anthropic_factory(model: Model) -> Provider:
        from .anthropic import AnthropicProvider

        return AnthropicProvider(api_key=model.api_key, base_url=model.base_url)

    def openai_factory(model: Model) -> Provider:
        from .openai import OpenAIProvider

        return OpenAIProvider(
            api_key=model.api_key,
            base_url=model.base_url,
            **model.config,
        )

    register_provider("anthropic-messages", anthropic_factory)
    register_provider("openai-completions", openai_factory)
    register_provider("openai-responses", openai_factory)


register_builtin_providers()
