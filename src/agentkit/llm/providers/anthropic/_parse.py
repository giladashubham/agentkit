from __future__ import annotations

from typing import Any

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

__all__ = ["parse_response", "map_stop_reason"]


def parse_response(response: Any) -> Response:
    content: list[Content] = []

    for block in response.content:
        if block.type == "text":
            content.append(TextContent(text=block.text))
        elif block.type == "thinking":
            content.append(ThinkingContent(text=block.thinking))
        elif block.type == "tool_use":
            content.append(ToolCall(
                id=block.id,
                name=block.name,
                arguments=block.input,
            ))

    return Response(
        message=Message(role=Role.ASSISTANT, content=content),
        stop_reason=map_stop_reason(response.stop_reason),
        usage=Usage(
            input=response.usage.input,
            output=response.usage.output,
            cache_read=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
            cache_write=getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
        ),
        model=response.model,
        raw=response,
    )


def map_stop_reason(reason: str) -> StopReason:
    mapping = {
        "end_turn": StopReason.STOP,
        "tool_use": StopReason.TOOL_USE,
        "max_tokens": StopReason.LENGTH,
        "stop_sequence": StopReason.STOP,
    }
    return mapping.get(reason, StopReason.STOP)
