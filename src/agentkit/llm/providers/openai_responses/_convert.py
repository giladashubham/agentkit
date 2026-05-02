from __future__ import annotations

from typing import Any

from ...context import Context
from ...providers.openai_ws._convert import (
    convert_context_to_input,
    convert_tool_choice,
    tool_to_responses_format,
)
from .._utils import timeout_seconds
from ..base import ModelOptions

__all__ = ["build_request"]


def build_request(context: Context, options: ModelOptions) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": options.model,
        "input": convert_context_to_input(context),
    }

    if context.system_prompt:
        request["instructions"] = context.system_prompt

    if options.max_tokens:
        request["max_output_tokens"] = options.max_tokens

    if options.temperature is not None:
        request["temperature"] = options.temperature

    if options.tools:
        request["tools"] = [tool_to_responses_format(t) for t in options.tools]

    if options.tool_choice:
        tool_choice = convert_tool_choice(options.tool_choice)
        if tool_choice is not None:
            request["tool_choice"] = tool_choice

    if options.timeout_ms is not None:
        request["timeout"] = timeout_seconds(options.timeout_ms)

    if options.headers:
        request["extra_headers"] = options.headers

    if options.reasoning:
        request["reasoning"] = {"effort": options.reasoning}

    request.update(options.extra)
    return request
