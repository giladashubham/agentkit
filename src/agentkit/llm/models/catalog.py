from __future__ import annotations

from ..model import Model, ModelCost
from .presets import openai_compatible_model

__all__ = ["BUILTIN_MODELS", "iter_builtin_models"]

# Static model catalog in a Pi-style data shape. Pricing is USD per 1M tokens.
# Keep this catalog intentionally curated; users can still register any custom Model.
BUILTIN_MODELS: dict[str, tuple[Model, ...]] = {
    "openai": (
        Model(
            provider="openai",
            api="openai-responses",
            id="gpt-4o-mini",
            name="GPT-4o mini",
            context_window=128_000,
            max_tokens=16_384,
            input_types=("text", "image"),
            cost=ModelCost(input=0.15, output=0.60, cache_read=0.075),
        ),
        Model(
            provider="openai",
            api="openai-responses",
            id="gpt-4o",
            name="GPT-4o",
            context_window=128_000,
            max_tokens=16_384,
            input_types=("text", "image"),
            cost=ModelCost(input=2.50, output=10.00, cache_read=1.25),
        ),
    ),
    "anthropic": (
        Model(
            provider="anthropic",
            api="anthropic-messages",
            id="claude-sonnet-4-20250514",
            name="Claude Sonnet 4",
            context_window=200_000,
            max_tokens=64_000,
            input_types=("text", "image"),
            reasoning=True,
            cost=ModelCost(input=3.00, output=15.00, cache_read=0.30, cache_write=3.75),
        ),
    ),
    "deepseek": (
        openai_compatible_model(
            "deepseek",
            "deepseek-chat",
            name="DeepSeek Chat",
            context_window=64_000,
            max_tokens=8_000,
        ),
    ),
    "groq": (
        openai_compatible_model(
            "groq",
            "llama-3.3-70b-versatile",
            name="Llama 3.3 70B Versatile",
            context_window=128_000,
            max_tokens=32_768,
        ),
    ),
    "openrouter": (
        openai_compatible_model(
            "openrouter",
            "openai/gpt-4o-mini",
            name="GPT-4o mini via OpenRouter",
            context_window=128_000,
            max_tokens=16_384,
            input_types=("text", "image"),
            cost=ModelCost(input=0.15, output=0.60),
        ),
    ),
    "xai": (
        openai_compatible_model(
            "xai",
            "grok-4",
            name="Grok 4",
            context_window=256_000,
            max_tokens=32_000,
            input_types=("text", "image"),
            reasoning=True,
        ),
    ),
    "fireworks": (
        openai_compatible_model(
            "fireworks",
            "accounts/fireworks/models/llama-v3p1-70b-instruct",
            name="Llama 3.1 70B Instruct",
            context_window=128_000,
            max_tokens=16_384,
        ),
    ),
    "together": (
        openai_compatible_model(
            "together",
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            name="Llama 3.3 70B Instruct Turbo",
            context_window=128_000,
            max_tokens=16_384,
        ),
    ),
    "ollama": (
        openai_compatible_model(
            "ollama",
            "llama3.2",
            name="Llama 3.2",
            context_window=128_000,
            max_tokens=8_192,
        ),
    ),
    "google": (
        Model(
            provider="google",
            api="google-generative-ai",
            id="gemini-2.5-flash",
            name="Gemini 2.5 Flash",
            context_window=1_000_000,
            max_tokens=65_536,
            input_types=("text", "image"),
            reasoning=True,
            cost=ModelCost(input=0.30, output=2.50),
        ),
        Model(
            provider="google",
            api="google-generative-ai",
            id="gemini-2.5-pro",
            name="Gemini 2.5 Pro",
            context_window=1_000_000,
            max_tokens=65_536,
            input_types=("text", "image"),
            reasoning=True,
            cost=ModelCost(input=1.25, output=10.00),
        ),
    ),
    "google-vertex": (
        Model(
            provider="google-vertex",
            api="google-vertex",
            id="gemini-2.5-flash",
            name="Gemini 2.5 Flash on Vertex AI",
            context_window=1_000_000,
            max_tokens=65_536,
            input_types=("text", "image"),
            reasoning=True,
            cost=ModelCost(input=0.30, output=2.50),
        ),
        Model(
            provider="google-vertex",
            api="google-vertex",
            id="gemini-2.5-pro",
            name="Gemini 2.5 Pro on Vertex AI",
            context_window=1_000_000,
            max_tokens=65_536,
            input_types=("text", "image"),
            reasoning=True,
            cost=ModelCost(input=1.25, output=10.00),
        ),
    ),
}


def iter_builtin_models() -> tuple[Model, ...]:
    """Return all built-in catalog models in provider order."""
    return tuple(model for models in BUILTIN_MODELS.values() for model in models)
