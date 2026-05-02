from __future__ import annotations

import json
from typing import Any

from ...types import Content, Message, Response, Role, StopReason, TextContent, ToolCall, Usage

__all__ = ["parse_response", "map_stop_reason"]


def parse_response(response: Any, model: str) -> Response:
    choice = response.choices[0]
    message = choice.message
    content: list[Content] = []

    if message.content:
        content.append(TextContent(text=message.content))

    if message.tool_calls:
        for tc in message.tool_calls:
            content.append(ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=json.loads(tc.function.arguments) if tc.function.arguments else {},
            ))

    return Response(
        message=Message(role=Role.ASSISTANT, content=content),
        stop_reason=map_stop_reason(choice.finish_reason),
        usage=Usage(
            input=response.usage.prompt_tokens if response.usage else 0,
            output=response.usage.completion_tokens if response.usage else 0,
        ),
        model=model,
        raw=response,
    )


def map_stop_reason(reason: str | None) -> StopReason:
    if reason is None:
        return StopReason.STOP
    mapping = {
        "stop": StopReason.STOP,
        "tool_calls": StopReason.TOOL_USE,
        "length": StopReason.LENGTH,
        "content_filter": StopReason.STOP,
    }
    return mapping.get(reason, StopReason.STOP)
