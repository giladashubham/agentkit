from __future__ import annotations

import json
from typing import Any

from ...context import Context
from ...tools import Tool
from ...types import ImageContent, Role, TextContent, ToolCall, ToolResult

__all__ = ["convert_context_to_input", "tool_to_responses_format", "convert_tool_choice"]


def convert_context_to_input(context: Context) -> list[dict[str, Any]]:
    """Convert our Context to the Responses API flat input list."""
    result: list[dict[str, Any]] = []

    for msg in context.messages:
        text_parts = [c for c in msg.content if isinstance(c, TextContent)]
        image_parts = [c for c in msg.content if isinstance(c, ImageContent)]
        tool_calls = [c for c in msg.content if isinstance(c, ToolCall)]
        tool_results = [c for c in msg.content if isinstance(c, ToolResult)]

        if msg.role == Role.TOOL_RESULT:
            for tr in tool_results:
                output = (
                    tr.content
                    if isinstance(tr.content, str)
                    else "".join(c.text for c in tr.content if isinstance(c, TextContent))
                )
                result.append({
                    "type": "function_call_output",
                    "call_id": tr.tool_call_id,
                    "output": output,
                })

        elif msg.role == Role.ASSISTANT:
            if text_parts:
                result.append({
                    "type": "message",
                    "role": "assistant",
                    "content": "".join(c.text for c in text_parts),
                })
            for tc in tool_calls:
                result.append({
                    "type": "function_call",
                    "call_id": tc.id,
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                })

        elif msg.role == Role.USER:
            if text_parts or image_parts:
                result.append({
                    "type": "message",
                    "role": "user",
                    "content": convert_message_content(text_parts, image_parts),
                })

    return result


def convert_message_content(
    text_parts: list[TextContent],
    image_parts: list[ImageContent],
) -> str | list[dict[str, str]]:
    text = "".join(c.text for c in text_parts)
    if not image_parts:
        return text

    content: list[dict[str, str]] = []
    if text:
        content.append({"type": "input_text", "text": text})
    for image in image_parts:
        content.append({
            "type": "input_image",
            "image_url": f"data:{image.mime_type};base64,{image.data}",
        })
    return content


def tool_to_responses_format(t: Tool) -> dict[str, Any]:
    """Convert our Tool to Responses API FunctionToolParam format."""
    return {
        "type": "function",
        "name": t.name,
        "description": t.description,
        "parameters": t.parameters,
        "strict": False,
    }


def convert_tool_choice(tool_choice: str | None) -> str | dict | None:
    if tool_choice is None or tool_choice == "auto":
        return "auto"
    if tool_choice == "any":
        return "required"
    if tool_choice == "none":
        return "none"
    return {"type": "function", "name": tool_choice}
