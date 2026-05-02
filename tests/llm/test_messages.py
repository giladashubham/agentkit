from __future__ import annotations

import pytest

from agentkit.llm import (
    AssistantMessage,
    Message,
    Role,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)


@pytest.mark.parametrize(
    ("factory", "role", "text"),
    [
        pytest.param(Message.user, Role.USER, "Hello", id="user"),
        pytest.param(Message.assistant, Role.ASSISTANT, "Hi there", id="assistant"),
    ],
)
def test_text_message_factories(factory, role: Role, text: str) -> None:
    msg = factory(text)

    assert msg.role == role
    assert msg.text() == text
    assert msg.tool_calls() == []
    assert msg.thinking() == ""
    assert msg.timestamp > 0
    assert isinstance(msg, UserMessage if role == Role.USER else AssistantMessage)


def test_explicit_tool_result_message() -> None:
    msg = Message.tool_result("call_1", "search", "result")

    assert isinstance(msg, ToolResultMessage)
    assert msg.role == Role.TOOL_RESULT
    assert msg.timestamp > 0


def test_assistant_message_with_text_thinking_and_tool_call() -> None:
    msg = Message(
        role=Role.ASSISTANT,
        content=[
            ThinkingContent(text="I should check."),
            TextContent(text="Let me check."),
            ToolCall(id="call_1", name="get_weather", arguments={"city": "Paris"}),
        ],
    )

    assert msg.text() == "Let me check."
    assert msg.thinking() == "I should check."
    assert msg.tool_calls() == [
        ToolCall(id="call_1", name="get_weather", arguments={"city": "Paris"})
    ]
