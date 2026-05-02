from __future__ import annotations

from .context import Context
from .model import Model, RunOptions
from .models.builtins import register_builtin_models
from .providers.base import ModelOptions
from .providers.builtins import register_builtin_providers
from .providers.registry import get_provider, list_provider_apis, register_provider
from .streaming import StreamResponse
from .types import Response


def _options(model: Model, context: Context, options: RunOptions | None) -> ModelOptions:
    options = options or RunOptions()
    headers = {**(model.headers or {}), **(options.headers or {})} or None
    max_tokens = options.max_tokens if options.max_tokens is not None else model.max_tokens or 4096
    return ModelOptions(
        model=model.id,
        max_tokens=max_tokens,
        temperature=options.temperature,
        top_p=options.top_p,
        stop_sequences=options.stop_sequences,
        tools=context.tools,
        tool_choice=options.tool_choice,
        reasoning=options.reasoning,
        reasoning_budget=options.reasoning_budget,
        transport=options.transport,
        timeout_ms=options.timeout_ms,
        max_retries=options.max_retries,
        headers=headers,
        abort_signal=options.abort_signal,
        on_payload=options.on_payload,
        on_response=options.on_response,
        model_ref=model,
        extra=dict(options.extra),
    )


async def complete(model: Model, context: Context, options: RunOptions | None = None) -> Response:
    """Run one non-streaming model request."""
    provider = get_provider(model)
    return await provider.complete(context, _options(model, context, options))


def stream(model: Model, context: Context, options: RunOptions | None = None) -> StreamResponse:
    """Run one streaming model request."""
    provider = get_provider(model)
    return provider.stream(context, _options(model, context, options))


register_builtin_models()
register_builtin_providers()

__all__ = [
    "complete",
    "stream",
    "get_provider",
    "list_provider_apis",
    "register_provider",
]
