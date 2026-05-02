"""LLM provider interfaces and optional provider implementations."""

from .base import ModelOptions, Provider
from .registry import get_provider, list_providers, register_provider

__all__ = [
    "ModelOptions",
    "Provider",
    "get_provider",
    "list_providers",
    "register_provider",
    "AnthropicProvider",
    "OpenAIProvider",
    "OpenAIWebSocketSession",
]


def __getattr__(name: str):
    if name == "AnthropicProvider":
        from .anthropic import AnthropicProvider

        return AnthropicProvider
    if name == "OpenAIProvider":
        from .openai import OpenAIProvider

        return OpenAIProvider
    if name == "OpenAIWebSocketSession":
        from .openai_ws import OpenAIWebSocketSession

        return OpenAIWebSocketSession
    raise AttributeError(name)
