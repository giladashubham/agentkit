from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "ProviderName",
    "ApiName",
    "ReasoningLevel",
    "Transport",
    "PayloadHook",
    "AsyncPayloadHook",
    "ResponseHook",
    "AsyncResponseHook",
    "Model",
    "RunOptions",
]

ProviderName = Literal[
    "anthropic",
    "openai",
    "google",
    "deepseek",
    "groq",
    "openrouter",
    "xai",
    "fireworks",
    "together",
    "ollama",
    "perplexity",
    "cerebras",
    "sambanova",
    "nebius",
]
ApiName = Literal[
    "anthropic-messages",
    "openai-completions",
    "openai-responses",
    "google-generative-ai",
    "google-vertex",
]
ReasoningLevel = Literal["minimal", "low", "medium", "high", "xhigh"]
Transport = Literal["sse", "websocket", "websocket-cached", "auto"]
PayloadHook = Callable[[dict[str, Any], Any], dict[str, Any] | None]
AsyncPayloadHook = Callable[[dict[str, Any], Any], Awaitable[dict[str, Any] | None]]
ResponseHook = Callable[[Any, Any], None]
AsyncResponseHook = Callable[[Any, Any], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class Model:
    """Model identity and provider/API configuration."""

    id: str
    provider: ProviderName | str
    api: ApiName | str
    name: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    headers: dict[str, str] | None = None
    context_window: int | None = None
    max_tokens: int | None = None
    input_types: tuple[Literal["text", "image"], ...] = ("text",)
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RunOptions:
    """Runtime options for one model request."""

    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop_sequences: list[str] | None = None
    tool_choice: str | None = None
    reasoning: ReasoningLevel | None = None
    reasoning_budget: int | None = None
    transport: Transport = "sse"
    timeout_ms: int | None = None
    max_retries: int | None = None
    headers: dict[str, str] | None = None
    abort_signal: asyncio.Event | None = None
    on_payload: PayloadHook | AsyncPayloadHook | None = None
    on_response: ResponseHook | AsyncResponseHook | None = None
    extra: dict[str, Any] = field(default_factory=dict)
