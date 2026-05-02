from __future__ import annotations

import os
from collections.abc import Mapping

__all__ = ["API_KEY_ENV_VARS", "get_env_api_key"]

API_KEY_ENV_VARS: dict[str, tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "google-vertex": ("GOOGLE_CLOUD_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "groq": ("GROQ_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
    "xai": ("XAI_API_KEY",),
    "fireworks": ("FIREWORKS_API_KEY",),
    "together": ("TOGETHER_API_KEY",),
    "perplexity": ("PERPLEXITY_API_KEY",),
    "cerebras": ("CEREBRAS_API_KEY",),
    "sambanova": ("SAMBANOVA_API_KEY",),
    "nebius": ("NEBIUS_API_KEY",),
}


def get_env_api_key(
    provider: str,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """Return the first configured API key for a known provider.

    Unknown providers return ``None``. Explicit ``Model.api_key`` values still
    take precedence; this helper is used as a provider-factory fallback.
    """
    env = os.environ if environ is None else environ
    for name in API_KEY_ENV_VARS.get(provider, ()):  # pragma: no branch - tiny loop
        value = env.get(name)
        if value:
            return value
    return None
