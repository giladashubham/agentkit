from __future__ import annotations

from typing import Any

from ...types import Content, Message, Response, Role, StopReason, TextContent, ToolCall, Usage

__all__ = ["parse_response", "map_stop_reason"]


def parse_response(response: Any, model: str) -> Response:
    content: list[Content] = []

    text = getattr(response, "text", None)
    if text:
        content.append(TextContent(text=text))

    for call in getattr(response, "function_calls", None) or []:
        content.append(ToolCall(
            id=getattr(call, "id", None) or getattr(call, "call_id", ""),
            name=getattr(call, "name", ""),
            arguments=dict(getattr(call, "args", {}) or {}),
        ))

    usage_metadata = getattr(response, "usage_metadata", None)
    usage = Usage(
        input=getattr(usage_metadata, "prompt_token_count", 0) if usage_metadata else 0,
        output=getattr(usage_metadata, "candidates_token_count", 0) if usage_metadata else 0,
    )

    finish_reason = None
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        finish_reason = getattr(candidates[0], "finish_reason", None)

    return Response(
        message=Message(role=Role.ASSISTANT, content=content),
        stop_reason=map_stop_reason(finish_reason),
        usage=usage,
        model=getattr(response, "model_version", None) or model,
        raw=response,
    )


def map_stop_reason(reason: Any) -> StopReason:
    value = getattr(reason, "name", reason)
    if value in {None, "STOP"}:
        return StopReason.STOP
    if value in {"MAX_TOKENS"}:
        return StopReason.LENGTH
    if value in {"MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL"}:
        return StopReason.ERROR
    return StopReason.STOP
