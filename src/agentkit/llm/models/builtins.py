from __future__ import annotations

from ..model import Model
from .presets import openai_compatible_model
from .registry import register_model

__all__ = ["register_builtin_models"]


def register_builtin_models() -> None:
    """Register a small built-in model set.

    This is intentionally tiny, not generated. Users can register their own models.
    """
    register_model(Model(provider="openai", api="openai-completions", id="gpt-4o-mini"))
    register_model(Model(provider="openai", api="openai-completions", id="gpt-4o"))
    register_model(Model(provider="openai", api="openai-responses", id="gpt-4o-mini"))
    register_model(Model(provider="openai", api="openai-responses", id="gpt-4o"))
    register_model(
        Model(provider="anthropic", api="anthropic-messages", id="claude-sonnet-4-20250514")
    )
    register_model(openai_compatible_model("deepseek", "deepseek-chat"))
    register_model(openai_compatible_model("groq", "llama-3.3-70b-versatile"))
    register_model(openai_compatible_model("openrouter", "openai/gpt-4o-mini"))
    register_model(openai_compatible_model("xai", "grok-4"))
    register_model(openai_compatible_model(
        "fireworks",
        "accounts/fireworks/models/llama-v3p1-70b-instruct",
    ))
    register_model(openai_compatible_model("together", "meta-llama/Llama-3.3-70B-Instruct-Turbo"))
    register_model(openai_compatible_model("ollama", "llama3.2"))
    register_model(Model(provider="google", api="google-generative-ai", id="gemini-2.5-flash"))
    register_model(Model(provider="google", api="google-generative-ai", id="gemini-2.5-pro"))
    register_model(Model(provider="google-vertex", api="google-vertex", id="gemini-2.5-flash"))
    register_model(Model(provider="google-vertex", api="google-vertex", id="gemini-2.5-pro"))
