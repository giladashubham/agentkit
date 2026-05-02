from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

from ..context import Context
from ..model import AsyncPayloadHook, AsyncResponseHook, PayloadHook, ResponseHook
from ..streaming import StreamEvent, StreamResponse
from ..tools import Tool
from ..types import Response


@dataclass(slots=True)
class ModelOptions:
    """Common options for all providers."""

    model: str
    max_tokens: int = 4096
    temperature: float | None = None
    top_p: float | None = None
    stop_sequences: list[str] | None = None
    tools: list[Tool] | None = None
    tool_choice: str | None = None  # "auto", "any", "none", or specific tool name
    reasoning: str | None = None
    reasoning_budget: int | None = None
    transport: Literal["sse", "websocket", "websocket-cached", "auto"] = "sse"
    timeout_ms: int | None = None
    max_retries: int | None = None
    headers: dict[str, str] | None = None
    abort_signal: asyncio.Event | None = None
    on_payload: PayloadHook | AsyncPayloadHook | None = None
    on_response: ResponseHook | AsyncResponseHook | None = None
    cache_control: Literal["none", "ephemeral"] = "none"
    model_ref: Any = None
    extra: dict[str, Any] = field(default_factory=dict)


class Provider(ABC):
    """Abstract base class for LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier."""
        ...

    @abstractmethod
    async def complete(
        self,
        context: Context,
        options: ModelOptions,
    ) -> Response:
        """Send a completion request and return the full response."""
        ...

    @abstractmethod
    def stream(
        self,
        context: Context,
        options: ModelOptions,
    ) -> StreamResponse:
        """Send a streaming request and return an async iterator of events."""
        ...

    @abstractmethod
    async def _stream_events(
        self,
        context: Context,
        options: ModelOptions,
    ) -> AsyncIterator[StreamEvent]:
        """Internal method to yield stream events."""
        ...
