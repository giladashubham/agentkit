from __future__ import annotations

from typing import Any

from ...context import Context
from ...types import Message, Role, TextContent, ToolCall, ToolResult
from ..base import ModelOptions

__all__ = ["build_request", "convert_messages", "convert_tool", "convert_tool_choice"]


def build_request(context: Context, options: ModelOptions, types: Any) -> dict[str, Any]:
    config: dict[str, Any] = {}

    if context.system_prompt:
        config["system_instruction"] = context.system_prompt

    if options.max_tokens:
        config["max_output_tokens"] = options.max_tokens

    if options.temperature is not None:
        config["temperature"] = options.temperature

    if options.top_p is not None:
        config["top_p"] = options.top_p

    if options.stop_sequences:
        config["stop_sequences"] = options.stop_sequences

    if options.tools:
        config["tools"] = [convert_tool(tool) for tool in options.tools]

    if options.tool_choice:
        tool_choice = convert_tool_choice(options.tool_choice)
        if tool_choice is not None:
            config["tool_config"] = tool_choice

    if options.reasoning:
        config["thinking_config"] = {"thinking_budget": options.reasoning_budget or -1}

    config.update(options.extra.pop("config", {}) if "config" in options.extra else {})

    request: dict[str, Any] = {
        "model": options.model,
        "contents": convert_messages(context.messages),
    }
    if config:
        request["config"] = types.GenerateContentConfig(**config)
    request.update(options.extra)
    return request


def convert_messages(messages: list[Message]) -> list[dict[str, Any]]:
    contents: list[dict[str, Any]] = []

    for msg in messages:
        parts: list[dict[str, Any]] = []
        for content in msg.content:
            if isinstance(content, TextContent):
                parts.append({"text": content.text})
            elif isinstance(content, ToolCall):
                parts.append({
                    "function_call": {
                        "id": content.id,
                        "name": content.name,
                        "args": content.arguments,
                    }
                })
            elif isinstance(content, ToolResult):
                output = (
                    content.content
                    if isinstance(content.content, str)
                    else "".join(c.text for c in content.content if isinstance(c, TextContent))
                )
                parts.append({
                    "function_response": {
                        "id": content.tool_call_id,
                        "name": content.tool_name,
                        "response": {"result": output, "is_error": content.is_error},
                    }
                })

        if not parts:
            continue

        role = "model" if msg.role == Role.ASSISTANT else "user"
        contents.append({"role": role, "parts": parts})

    return contents


def convert_tool(tool: Any) -> dict[str, Any]:
    return {
        "function_declarations": [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
        ]
    }


def convert_tool_choice(tool_choice: str | None) -> dict[str, Any] | None:
    if tool_choice is None or tool_choice == "auto":
        return {"function_calling_config": {"mode": "AUTO"}}
    if tool_choice == "any":
        return {"function_calling_config": {"mode": "ANY"}}
    if tool_choice == "none":
        return {"function_calling_config": {"mode": "NONE"}}
    return {
        "function_calling_config": {
            "mode": "ANY",
            "allowed_function_names": [tool_choice],
        }
    }
