from __future__ import annotations

import json
from typing import Any

from ...context import Context
from ...types import Content, ImageContent, Message, Role, TextContent, ToolCall, ToolResult
from .._utils import timeout_seconds
from ..base import ModelOptions

__all__ = ["build_request", "convert_messages", "convert_content"]


def build_request(context: Context, options: ModelOptions) -> dict[str, Any]:
    messages = convert_messages(context.messages)

    if context.system_prompt:
        messages.insert(0, {"role": "system", "content": context.system_prompt})

    request: dict[str, Any] = {
        "model": options.model,
        "messages": messages,
    }

    if options.max_tokens:
        request["max_completion_tokens"] = options.max_tokens

    if options.temperature is not None:
        request["temperature"] = options.temperature

    if options.top_p is not None:
        request["top_p"] = options.top_p

    if options.stop_sequences:
        request["stop"] = options.stop_sequences

    if options.tools:
        request["tools"] = [t.to_openai() for t in options.tools]

    if options.tool_choice:
        if options.tool_choice == "auto":
            request["tool_choice"] = "auto"
        elif options.tool_choice == "any":
            request["tool_choice"] = "required"
        elif options.tool_choice == "none":
            request["tool_choice"] = "none"
        else:
            request["tool_choice"] = {
                "type": "function",
                "function": {"name": options.tool_choice},
            }

    if options.reasoning:
        request["reasoning_effort"] = options.reasoning

    if options.timeout_ms is not None:
        request["timeout"] = timeout_seconds(options.timeout_ms)

    if options.headers:
        request["extra_headers"] = options.headers

    request.update(options.extra)
    return request


def convert_messages(messages: list[Message]) -> list[dict[str, Any]]:
    result = []
    for msg in messages:
        if msg.role == Role.TOOL_RESULT:
            tool_results = [c for c in msg.content if isinstance(c, ToolResult)]
            for tr in tool_results:
                result.append({
                    "role": "tool",
                    "tool_call_id": tr.tool_call_id,
                    "content": tr.content if isinstance(tr.content, str) else str(tr.content),
                })

        elif msg.role == Role.USER:
            content = convert_content(msg.content)
            result.append({"role": "user", "content": content})

        elif msg.role == Role.ASSISTANT:
            tool_calls = [c for c in msg.content if isinstance(c, ToolCall)]
            text_content = [c for c in msg.content if isinstance(c, TextContent)]

            assistant_msg: dict[str, Any] = {"role": "assistant"}

            if text_content:
                assistant_msg["content"] = "".join(c.text for c in text_content)

            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in tool_calls
                ]

            result.append(assistant_msg)

    return result


def convert_content(content: list[Content]) -> str | list[dict[str, Any]]:
    if len(content) == 1 and isinstance(content[0], TextContent):
        return content[0].text

    result = []
    for c in content:
        if isinstance(c, TextContent):
            result.append({"type": "text", "text": c.text})
        elif isinstance(c, ImageContent):
            result.append({
                "type": "image_url",
                "image_url": {"url": f"data:{c.mime_type};base64,{c.data}"},
            })
    return result if result else ""
