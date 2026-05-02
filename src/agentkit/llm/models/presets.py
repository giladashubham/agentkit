from __future__ import annotations

from dataclasses import replace
from typing import Literal

from ..model import Model, ModelCost

__all__ = ["openai_compatible_model", "OPENAI_COMPATIBLE_BASE_URLS"]

OPENAI_COMPATIBLE_BASE_URLS: dict[str, str] = {
    "deepseek": "https://api.deepseek.com",
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "xai": "https://api.x.ai/v1",
    "fireworks": "https://api.fireworks.ai/inference/v1",
    "together": "https://api.together.xyz/v1",
    "ollama": "http://localhost:11434/v1",
    "perplexity": "https://api.perplexity.ai",
    "cerebras": "https://api.cerebras.ai/v1",
    "sambanova": "https://api.sambanova.ai/v1",
    "nebius": "https://api.studio.nebius.ai/v1",
}


def openai_compatible_model(
    provider: str,
    id: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    api: Literal["openai-completions", "openai-responses"] = "openai-completions",
    name: str | None = None,
    headers: dict[str, str] | None = None,
    context_window: int | None = None,
    max_tokens: int | None = None,
    input_types: tuple[Literal["text", "image"], ...] = ("text",),
    reasoning: bool = False,
    cost: ModelCost | None = None,
    config: dict | None = None,
) -> Model:
    """Create a Model for an OpenAI-compatible provider preset.

    Unknown providers are allowed if `base_url` is supplied.
    """
    resolved_base_url = base_url or OPENAI_COMPATIBLE_BASE_URLS.get(provider)
    if resolved_base_url is None:
        available = ", ".join(sorted(OPENAI_COMPATIBLE_BASE_URLS))
        raise ValueError(
            f"Unknown OpenAI-compatible provider: {provider}. "
            f"Pass base_url or use one of: {available}"
        )

    return Model(
        provider=provider,
        api=api,
        id=id,
        name=name,
        base_url=resolved_base_url,
        api_key=api_key,
        headers=headers,
        context_window=context_window,
        max_tokens=max_tokens,
        input_types=input_types,
        reasoning=reasoning,
        cost=cost or ModelCost(),
        config=config or {},
    )


def with_openai_compatible_base_url(model: Model) -> Model:
    """Return model with a preset base_url when its provider is known."""
    if model.base_url is not None:
        return model
    base_url = OPENAI_COMPATIBLE_BASE_URLS.get(model.provider)
    return replace(model, base_url=base_url) if base_url else model
