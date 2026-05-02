from __future__ import annotations

from ..env import get_env_api_key
from ..model import Model
from ..models.presets import with_openai_compatible_base_url
from .base import Provider
from .registry import register_provider

__all__ = ["register_builtin_providers"]


def _api_key_for_model(model: Model) -> str | None:
    return model.api_key or get_env_api_key(model.provider)


def _create_anthropic(model: Model) -> Provider:
    from .anthropic import AnthropicProvider

    return AnthropicProvider(api_key=_api_key_for_model(model), base_url=model.base_url)


def _create_openai(model: Model) -> Provider:
    from .openai import OpenAIProvider

    model = with_openai_compatible_base_url(model)
    return OpenAIProvider(
        api_key=_api_key_for_model(model),
        base_url=model.base_url,
        **model.config,
    )


def _create_openai_responses(model: Model) -> Provider:
    from .openai_responses import OpenAIResponsesProvider

    model = with_openai_compatible_base_url(model)
    return OpenAIResponsesProvider(
        api_key=_api_key_for_model(model),
        base_url=model.base_url,
        **model.config,
    )


def _create_google(model: Model) -> Provider:
    from .google import GoogleProvider

    return GoogleProvider(
        api_key=_api_key_for_model(model),
        base_url=model.base_url,
        vertexai=model.api == "google-vertex",
        **model.config,
    )


def register_builtin_providers() -> None:
    """Register built-in API provider factories."""
    register_provider("anthropic-messages", _create_anthropic)
    register_provider("openai-completions", _create_openai)
    register_provider("openai-responses", _create_openai_responses)
    register_provider("google-generative-ai", _create_google)
    register_provider("google-vertex", _create_google)
