from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

try:
    from anthropic import AsyncAnthropic
except ImportError as exc:  # pragma: no cover - depends on optional extra
    from agentkit.exceptions import ProviderDependencyError

    raise ProviderDependencyError(
        "Anthropic support requires the 'anthropic' extra. "
        "Install it with: pip install 'agentkit[anthropic]'"
    ) from exc

from ..._hooks import apply_payload_hook, apply_response_hook, check_abort
from ...context import Context
from ...streaming import StreamEvent, StreamResponse
from ...types import (
    Content,
    Message,
    Response,
    Role,
    StopReason,
    TextContent,
    ThinkingContent,
    ToolCall,
    Usage,
)
from ..base import ModelOptions, Provider
from ._convert import build_request
from ._parse import map_stop_reason, parse_response

__all__ = ["AnthropicProvider"]


class AnthropicProvider(Provider):
    """Anthropic Claude provider."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self._client = AsyncAnthropic(api_key=api_key, base_url=base_url)

    @property
    def name(self) -> str:
        return "anthropic"

    async def complete(self, context: Context, options: ModelOptions) -> Response:
        request = await apply_payload_hook(build_request(context, options), options)
        response = await self._client.messages.create(**request, stream=False)
        await apply_response_hook(response, options)
        return parse_response(response)

    def stream(self, context: Context, options: ModelOptions) -> StreamResponse:
        return StreamResponse(self._stream_events(context, options))

    async def _stream_events(
        self,
        context: Context,
        options: ModelOptions,
    ) -> AsyncIterator[StreamEvent]:
        accumulated_text = ""
        accumulated_thinking = ""
        tool_calls: dict[int, dict[str, Any]] = {}
        usage = Usage()
        stop_reason = StopReason.STOP
        model = options.model

        request = await apply_payload_hook(build_request(context, options), options)
        async with self._client.messages.stream(**request) as stream:
            yield StreamEvent.start(model)

            async for event in stream:
                check_abort(options)
                if event.type == "message_start":
                    if hasattr(event, "message") and hasattr(event.message, "usage"):
                        usage.input = event.message.usage.input

                elif event.type == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        tool_calls[event.index] = {
                            "id": block.id,
                            "name": block.name,
                            "arguments_json": "",
                        }
                        yield StreamEvent.toolcall_start(block.id, block.name)

                elif event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        accumulated_text += delta.text
                        yield StreamEvent.text_delta(delta.text)
                    elif delta.type == "thinking_delta":
                        accumulated_thinking += delta.thinking
                        yield StreamEvent.thinking_delta(delta.thinking)
                    elif delta.type == "input_json_delta":
                        if event.index in tool_calls:
                            tool_calls[event.index]["arguments_json"] += delta.partial_json
                            yield StreamEvent.toolcall_delta(
                                tool_calls[event.index]["id"],
                                delta.partial_json,
                            )

                elif event.type == "content_block_stop":
                    if event.index in tool_calls:
                        tc = tool_calls[event.index]
                        try:
                            arguments = (
                                json.loads(tc["arguments_json"]) if tc["arguments_json"] else {}
                            )
                        except json.JSONDecodeError:
                            arguments = {}
                        yield StreamEvent.toolcall_end(tc["id"], tc["name"], arguments)

                elif event.type == "message_delta":
                    if hasattr(event, "delta"):
                        if hasattr(event.delta, "stop_reason") and event.delta.stop_reason:
                            stop_reason = map_stop_reason(event.delta.stop_reason)
                    if hasattr(event, "usage"):
                        usage.output = event.usage.output

            if accumulated_text:
                yield StreamEvent.text_end(accumulated_text)
            if accumulated_thinking:
                yield StreamEvent.thinking_end(accumulated_thinking)

            content: list[Content] = []
            if accumulated_thinking:
                content.append(ThinkingContent(text=accumulated_thinking))
            if accumulated_text:
                content.append(TextContent(text=accumulated_text))
            for tc in tool_calls.values():
                try:
                    arguments = json.loads(tc["arguments_json"]) if tc["arguments_json"] else {}
                except json.JSONDecodeError:
                    arguments = {}
                content.append(ToolCall(id=tc["id"], name=tc["name"], arguments=arguments))

            response = Response(
                message=Message(role=Role.ASSISTANT, content=content),
                stop_reason=stop_reason,
                usage=usage,
                model=model,
            )
            yield StreamEvent.done(response)
