from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .types import AssistantMessage, Message, TextContent, ThinkingContent, ToolCall

__all__ = ["EventType", "StreamEvent", "StreamResponse"]


class EventType(StrEnum):
    START = "start"
    TEXT_START = "text_start"
    TEXT_DELTA = "text_delta"
    TEXT_END = "text_end"
    THINKING_START = "thinking_start"
    THINKING_DELTA = "thinking_delta"
    THINKING_END = "thinking_end"
    TOOLCALL_START = "toolcall_start"
    TOOLCALL_DELTA = "toolcall_delta"
    TOOLCALL_END = "toolcall_end"
    DONE = "done"
    ERROR = "error"


@dataclass(slots=True)
class StreamEvent:
    type: EventType
    data: Any = None
    partial: Message | None = None
    content_index: int | None = None

    @classmethod
    def start(cls, model: str, partial: Message | None = None) -> StreamEvent:
        return cls(
            type=EventType.START,
            data={"model": model},
            partial=partial or AssistantMessage(content=[]),
        )

    @classmethod
    def text_start(cls, content_index: int = 0, partial: Message | None = None) -> StreamEvent:
        return cls(
            type=EventType.TEXT_START,
            partial=partial or AssistantMessage(content=[]),
            content_index=content_index,
        )

    @classmethod
    def text_delta(
        cls,
        delta: str,
        content_index: int = 0,
        partial: Message | None = None,
    ) -> StreamEvent:
        return cls(
            type=EventType.TEXT_DELTA,
            data=delta,
            partial=partial or AssistantMessage(content=[TextContent(text=delta)]),
            content_index=content_index,
        )

    @classmethod
    def text_end(
        cls,
        text: str,
        content_index: int = 0,
        partial: Message | None = None,
    ) -> StreamEvent:
        return cls(
            type=EventType.TEXT_END,
            data=text,
            partial=partial or AssistantMessage(content=[TextContent(text=text)]),
            content_index=content_index,
        )

    @classmethod
    def thinking_start(cls, content_index: int = 0, partial: Message | None = None) -> StreamEvent:
        return cls(
            type=EventType.THINKING_START,
            partial=partial or AssistantMessage(content=[]),
            content_index=content_index,
        )

    @classmethod
    def thinking_delta(
        cls,
        delta: str,
        content_index: int = 0,
        partial: Message | None = None,
    ) -> StreamEvent:
        return cls(
            type=EventType.THINKING_DELTA,
            data=delta,
            partial=partial or AssistantMessage(content=[ThinkingContent(text=delta)]),
            content_index=content_index,
        )

    @classmethod
    def thinking_end(
        cls,
        text: str,
        content_index: int = 0,
        partial: Message | None = None,
    ) -> StreamEvent:
        return cls(
            type=EventType.THINKING_END,
            data=text,
            partial=partial or AssistantMessage(content=[ThinkingContent(text=text)]),
            content_index=content_index,
        )

    @classmethod
    def toolcall_start(
        cls,
        id: str = "",
        name: str = "",
        content_index: int = 0,
        partial: Message | None = None,
    ) -> StreamEvent:
        return cls(
            type=EventType.TOOLCALL_START,
            partial=partial or AssistantMessage(content=[ToolCall(id=id, name=name, arguments={})]),
            content_index=content_index,
        )

    @classmethod
    def toolcall_delta(
        cls,
        id: str,
        delta: str,
        content_index: int = 0,
        partial: Message | None = None,
    ) -> StreamEvent:
        return cls(
            type=EventType.TOOLCALL_DELTA,
            data=delta,
            partial=partial or AssistantMessage(content=[ToolCall(id=id, name="", arguments={})]),
            content_index=content_index,
        )

    @classmethod
    def toolcall_end(
        cls,
        tool_call: ToolCall | str,
        name: str | None = None,
        arguments: dict[str, Any] | None = None,
        content_index: int = 0,
        partial: Message | None = None,
    ) -> StreamEvent:
        if isinstance(tool_call, str):
            tool_call = ToolCall(id=tool_call, name=name or "", arguments=arguments or {})
        return cls(
            type=EventType.TOOLCALL_END,
            data=tool_call,
            partial=partial or AssistantMessage(content=[tool_call]),
            content_index=content_index,
        )

    @classmethod
    def done(cls, response: Any) -> StreamEvent:
        return cls(type=EventType.DONE, data=response, partial=response.message)

    @classmethod
    def error(cls, error: Exception, partial: Message | None = None) -> StreamEvent:
        return cls(type=EventType.ERROR, data=error, partial=partial)


class StreamResponse:
    """Async iterator over stream events with final response access."""

    def __init__(self, stream: AsyncIterator[StreamEvent]):
        self._stream = stream
        self._response: Any = None
        self._text_chunks: list[str] = []
        self._thinking_chunks: list[str] = []

    def __aiter__(self) -> AsyncIterator[StreamEvent]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[StreamEvent]:
        async for event in self._stream:
            if event.type == EventType.TEXT_DELTA:
                self._text_chunks.append(event.data)
            elif event.type == EventType.THINKING_DELTA:
                self._thinking_chunks.append(event.data)
            elif event.type == EventType.DONE:
                self._response = event.data

            yield event

    async def result(self) -> Any:
        """Consume stream and return the final response."""
        async for _ in self:
            pass
        return self._response

    async def text(self) -> str:
        """Consume stream and return accumulated text."""
        await self.result()
        return "".join(self._text_chunks)

    async def collect_text(self) -> AsyncIterator[str]:
        """Yield text deltas as they arrive."""
        async for event in self:
            if event.type == EventType.TEXT_DELTA:
                yield event.data
