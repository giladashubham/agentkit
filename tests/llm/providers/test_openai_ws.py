"""Tests for WebSocket session helpers — no live connection needed."""

import json

import pytest

pytest.importorskip("openai")

from agentkit.llm import Context, Role, TextContent, ToolCall, ToolResult, tool
from agentkit.llm.providers.openai_ws._convert import (
    convert_context_to_input,
    convert_tool_choice,
    tool_to_responses_format,
)


def test_convert_user_text():
    ctx = Context()
    ctx.add_user("Hello world")
    items = convert_context_to_input(ctx)
    assert items == [{"type": "message", "role": "user", "content": "Hello world"}]


def test_convert_assistant_text():
    ctx = Context()
    ctx.add_assistant("Hi there")
    items = convert_context_to_input(ctx)
    assert items == [{"type": "message", "role": "assistant", "content": "Hi there"}]


def test_convert_tool_call():
    from agentkit.llm.types import Message
    ctx = Context()
    ctx.add_message(Message(
        role=Role.ASSISTANT,
        content=[ToolCall(id="call_123", name="search", arguments={"query": "AI"})],
    ))
    items = convert_context_to_input(ctx)
    assert len(items) == 1
    assert items[0]["type"] == "function_call"
    assert items[0]["call_id"] == "call_123"
    assert items[0]["name"] == "search"
    assert json.loads(items[0]["arguments"]) == {"query": "AI"}


def test_convert_tool_result():
    from agentkit.llm.types import Message
    ctx = Context()
    ctx.add_message(Message(
        role=Role.TOOL_RESULT,
        content=[
            ToolResult(
                tool_call_id="call_123",
                tool_name="search",
                content="Search results here",
            )
        ],
    ))
    items = convert_context_to_input(ctx)
    assert len(items) == 1
    assert items[0]["type"] == "function_call_output"
    assert items[0]["call_id"] == "call_123"
    assert items[0]["output"] == "Search results here"


def test_convert_mixed_turn():
    """Full agent turn: user → assistant text + tool call → tool result."""
    from agentkit.llm.types import Message
    ctx = Context(system_prompt="You are helpful")
    ctx.add_user("Search for Python tutorials")
    ctx.add_message(Message(
        role=Role.ASSISTANT,
        content=[
            TextContent(text="I'll search for that."),
            ToolCall(id="call_456", name="search", arguments={"query": "Python tutorials"}),
        ],
    ))
    ctx.add_message(Message(
        role=Role.TOOL_RESULT,
        content=[
            ToolResult(
                tool_call_id="call_456",
                tool_name="search",
                content="Found 10 results",
            )
        ],
    ))

    items = convert_context_to_input(ctx)
    assert len(items) == 4
    assert items[0]["role"] == "user"
    assert items[1]["role"] == "assistant"
    assert items[2]["type"] == "function_call"
    assert items[3]["type"] == "function_call_output"
    assert items[3]["call_id"] == "call_456"


def testconvert_tool_choice():
    assert convert_tool_choice("auto") == "auto"
    assert convert_tool_choice(None) == "auto"
    assert convert_tool_choice("any") == "required"
    assert convert_tool_choice("none") == "none"
    assert convert_tool_choice("search") == {"type": "function", "name": "search"}


def testtool_to_responses_format():
    @tool()
    def search(query: str, max_results: int = 5) -> str:
        """Search the web for information."""
        return "results"

    fmt = tool_to_responses_format(search)
    assert fmt["type"] == "function"
    assert fmt["name"] == "search"
    assert fmt["description"] == "Search the web for information."
    assert "query" in fmt["parameters"]["properties"]
    assert fmt["strict"] is False
