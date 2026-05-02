from __future__ import annotations

import json
from typing import Any

from ...types import Content, Message, Response, Role, StopReason, TextContent, ToolCall, Usage

__all__ = ["parse_response", "map_stop_reason", "response_text"]


def response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text

    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for part in getattr(item, "content", []) or []:
            value = getattr(part, "text", None)
            if value:
                chunks.append(value)
    return "".join(chunks)


def parse_response(response: Any, model: str | None = None) -> Response:
    content: list[Content] = []
    text = response_text(response)
    if text:
        content.append(TextContent(text=text))

    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) == "function_call":
            raw_arguments = getattr(item, "arguments", "") or "{}"
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                arguments = {}
            content.append(ToolCall(
                id=getattr(item, "call_id", None) or getattr(item, "id", ""),
                name=getattr(item, "name", ""),
                arguments=arguments,
            ))

    usage_data = getattr(response, "usage", None)
    cache_read = 0
    if usage_data:
        details = getattr(usage_data, "input_tokens_details", None)
        if details:
            cache_read = getattr(details, "cached_tokens", 0) or 0

    return Response(
        message=Message(role=Role.ASSISTANT, content=content),
        stop_reason=map_stop_reason(getattr(response, "status", None)),
        usage=Usage(
            input=getattr(usage_data, "input", 0) or getattr(usage_data, "input_tokens", 0)
            if usage_data
            else 0,
            output=getattr(usage_data, "output", 0) or getattr(usage_data, "output_tokens", 0)
            if usage_data
            else 0,
            cache_read=cache_read,
        ),
        model=getattr(response, "model", None) or model or "",
        raw=response,
    )


def map_stop_reason(reason: str | None) -> StopReason:
    if reason == "incomplete":
        return StopReason.LENGTH
    if reason == "failed":
        return StopReason.ERROR
    if reason == "cancelled":
        return StopReason.ABORTED
    return StopReason.STOP
