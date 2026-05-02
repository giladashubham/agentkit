from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("openai")

from agentkit.llm import Context
from agentkit.llm.providers.base import ModelOptions
from agentkit.llm.providers.openai_responses._convert import build_request
from agentkit.llm.providers.openai_responses._parse import parse_response


def test_openai_responses_build_request() -> None:
    ctx = Context(system_prompt="You are helpful.")
    ctx.add_user("hello")

    request = build_request(ctx, ModelOptions(model="gpt-4o-mini", max_tokens=100))

    assert request["model"] == "gpt-4o-mini"
    assert request["instructions"] == "You are helpful."
    assert request["max_output_tokens"] == 100
    assert request["input"] == [{"type": "message", "role": "user", "content": "hello"}]


def test_openai_responses_parse_response_text_and_tool_call() -> None:
    response = SimpleNamespace(
        output_text="hello",
        output=[
            SimpleNamespace(
                type="function_call",
                call_id="call_1",
                name="search",
                arguments='{"q":"x"}',
            )
        ],
        usage=SimpleNamespace(input_tokens=3, output_tokens=4),
        status="completed",
        model="gpt-4o-mini",
    )

    parsed = parse_response(response)

    assert parsed.text() == "hello"
    assert parsed.tool_calls()[0].name == "search"
    assert parsed.usage.input == 3
    assert parsed.usage.output == 4
