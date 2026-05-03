from __future__ import annotations

from typing import Any

from ...context import Context
from ...types import ImageContent, Message, Role, TextContent, ThinkingContent, ToolCall, ToolResult
from .._utils import timeout_seconds
from ..base import ModelOptions

__all__ = ["build_request", "convert_messages", "reasoning_budget"]


def reasoning_budget(level: str) -> int:
    return {
        "minimal": 1024,
        "low": 4096,
        "medium": 8192,
        "high": 16000,
        "xhigh": 32000,
    }.get(level, 8192)


def build_request(context: Context, options: ModelOptions) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": options.model,
        "max_tokens": options.max_tokens,
        "messages": convert_messages(context.messages),
    }

    if context.system_prompt:
        if options.cache_control == "ephemeral":
            request["system"] = [
                {
                    "type": "text",
                    "text": context.system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        else:
            request["system"] = context.system_prompt

    if options.temperature is not None:
        request["temperature"] = options.temperature

    if options.top_p is not None:
        request["top_p"] = options.top_p

    if options.stop_sequences:
        request["stop_sequences"] = options.stop_sequences

    if options.tools:
        tools = [t.to_anthropic() for t in options.tools]
        if options.cache_control == "ephemeral":
            tools[-1] = {**tools[-1], "cache_control": {"type": "ephemeral"}}
        request["tools"] = tools

    if options.tool_choice:
        if options.tool_choice == "auto":
            request["tool_choice"] = {"type": "auto"}
        elif options.tool_choice == "any":
            request["tool_choice"] = {"type": "any"}
        elif options.tool_choice == "none":
            pass
        else:
            request["tool_choice"] = {"type": "tool", "name": options.tool_choice}

    if options.reasoning:
        request["thinking"] = {
            "type": "enabled",
            "budget_tokens": options.reasoning_budget or reasoning_budget(options.reasoning),
        }
        request["temperature"] = 1

    if options.timeout_ms is not None:
        request["timeout"] = timeout_seconds(options.timeout_ms)

    if options.headers:
        request["extra_headers"] = options.headers

    request.update(options.extra)
    return request


def convert_messages(messages: list[Message]) -> list[dict[str, Any]]:
    result = []
    for msg in messages:
        content = []
        for c in msg.content:
            if isinstance(c, TextContent):
                content.append({"type": "text", "text": c.text})
            elif isinstance(c, ImageContent):
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": c.mime_type,
                        "data": c.data,
                    },
                })
            elif isinstance(c, ToolCall):
                content.append({
                    "type": "tool_use",
                    "id": c.id,
                    "name": c.name,
                    "input": c.arguments,
                })
            elif isinstance(c, ToolResult):
                content.append({
                    "type": "tool_result",
                    "tool_use_id": c.tool_call_id,
                    "content": c.content
                    if isinstance(c.content, str)
                    else convert_tool_result_content(c.content),
                    "is_error": c.is_error,
                })
            elif isinstance(c, ThinkingContent):
                if c.redacted:
                    content.append({"type": "redacted_thinking", "data": c.signature or ""})
                else:
                    block: dict[str, Any] = {"type": "thinking", "thinking": c.text}
                    if c.signature:
                        block["signature"] = c.signature
                    content.append(block)

        result.append({
            "role": "user" if msg.role == Role.TOOL_RESULT else msg.role.value,
            "content": content,
        })
    return result


def convert_tool_result_content(content: list[TextContent | ImageContent]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in content:
        if isinstance(item, TextContent):
            result.append({"type": "text", "text": item.text})
        elif isinstance(item, ImageContent):
            result.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": item.mime_type,
                    "data": item.data,
                },
            })
    return result
