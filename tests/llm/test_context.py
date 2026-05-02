from __future__ import annotations

from agentkit.llm import Context, Message, Role, ThinkingContent, ToolCall


def test_context_round_trip_preserves_messages_system_and_metadata() -> None:
    ctx = Context(system_prompt="You are helpful", metadata={"thread_id": "abc"})
    ctx.add_user("Hello")
    ctx.add_assistant("Hi!")
    ctx.add_message(
        Message(
            role=Role.ASSISTANT,
            content=[
                ThinkingContent(text="Need a tool."),
                ToolCall(id="call_1", name="search", arguments={"query": "agentkit"}),
            ],
        )
    )
    ctx.add_tool_result("call_1", "search", "Search results")

    data = ctx.to_dict()
    restored = Context.from_dict(data)

    assert data["systemPrompt"] == "You are helpful"
    assert data["messages"][2]["content"][1]["type"] == "toolCall"
    assert data["messages"][3]["role"] == "toolResult"
    assert data["messages"][3]["toolCallId"] == "call_1"
    assert data["messages"][3]["toolName"] == "search"
    assert data["messages"][3]["isError"] is False
    assert restored.system_prompt == "You are helpful"
    assert restored.metadata == {"thread_id": "abc"}
    assert restored.messages[-1].role == Role.TOOL_RESULT
    assert restored.messages[0].timestamp == ctx.messages[0].timestamp
    assert restored.to_dict() == ctx.to_dict()


def test_context_copy_is_independent() -> None:
    ctx = Context(system_prompt="system").add_user("hello")
    copied = ctx.copy()

    copied.system_prompt = "new system"
    copied.add_assistant("hi")

    assert ctx.system_prompt == "system"
    assert [m.text() for m in ctx.messages] == ["hello"]
    assert [m.text() for m in copied.messages] == ["hello", "hi"]


def test_context_clear_removes_messages_only() -> None:
    ctx = Context(system_prompt="system", metadata={"key": "value"}).add_user("hello")

    result = ctx.clear()

    assert result is ctx
    assert ctx.messages == []
    assert ctx.system_prompt == "system"
    assert ctx.metadata == {"key": "value"}
