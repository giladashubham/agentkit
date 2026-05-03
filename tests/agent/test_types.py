from __future__ import annotations

from agentkit.agent import (
    AfterToolCallContext,
    AgentEvent,
    AgentEventType,
    AgentToolResult,
    BeforeToolCallContext,
    BeforeToolCallResult,
    ShouldStopAfterTurnContext,
)
from agentkit.llm import Message, ToolCall


def test_agent_event_factories() -> None:
    message = Message.assistant("hello")
    tool_call = ToolCall(id="tc", name="echo", arguments={})
    result = AgentToolResult("ok")

    assert AgentEvent.agent_start().type == AgentEventType.AGENT_START
    assert AgentEvent.agent_end([]).type == AgentEventType.AGENT_END
    assert AgentEvent.agent_end([]).data == []
    assert AgentEvent.turn_start(1).data == 1
    assert AgentEvent.turn_end(1, message, []).data == (1, message, [])
    assert AgentEvent.message_start(message).data is message
    assert AgentEvent.message_update(message).data == (message, None)
    assert AgentEvent.message_end(message).data is message
    assert AgentEvent.tool_execution_start(tool_call).data is tool_call
    assert AgentEvent.tool_execution_update(tool_call, "u").data == (tool_call, "u")
    assert AgentEvent.tool_execution_end(tool_call, result).data == (tool_call, result, False)
    assert AgentEvent.tool_execution_end(tool_call, result, True).data == (tool_call, result, True)


def test_agent_tool_result_defaults() -> None:
    result = AgentToolResult("ok")

    assert result.content == "ok"
    assert result.details is None
    assert result.terminate is False


def test_before_tool_call_result_defaults() -> None:
    r = BeforeToolCallResult("blocked")

    assert r.content == "blocked"
    assert r.is_error is False
    assert r.terminate is False


def test_before_tool_call_context_fields(echo_agent_tool, mock_model) -> None:  # type: ignore[no-untyped-def]
    from agentkit.agent import AgentState

    msg = Message.assistant("hi")
    tc = ToolCall(id="1", name="echo", arguments={})
    args = {"value": "hi"}
    state = AgentState(mock_model)
    ctx = BeforeToolCallContext(
        tool_call=tc,
        agent_tool=echo_agent_tool,
        assistant_message=msg,
        arguments=args,
        state=state,
    )

    assert ctx.tool_call is tc
    assert ctx.agent_tool is echo_agent_tool
    assert ctx.assistant_message is msg
    assert ctx.arguments is args
    assert ctx.args is args
    assert ctx.state is state


def test_after_tool_call_context_fields(echo_agent_tool, mock_model) -> None:  # type: ignore[no-untyped-def]
    from agentkit.agent import AgentState

    msg = Message.assistant("hi")
    tc = ToolCall(id="1", name="echo", arguments={})
    result = AgentToolResult("ok")
    args = {"value": "hi"}
    state = AgentState(mock_model)
    ctx = AfterToolCallContext(
        tool_call=tc,
        agent_tool=echo_agent_tool,
        result=result,
        is_error=False,
        assistant_message=msg,
        arguments=args,
        state=state,
    )

    assert ctx.result is result
    assert ctx.is_error is False
    assert ctx.arguments is args
    assert ctx.args is args
    assert ctx.state is state


def test_should_stop_after_turn_context_aliases(mock_model) -> None:  # type: ignore[no-untyped-def]
    from agentkit.agent import AgentState

    msg = Message.assistant("hi")
    state = AgentState(mock_model)
    ctx = ShouldStopAfterTurnContext(
        assistant_message=msg,
        tool_result_messages=[],
        state=state,
        new_messages=[msg],
    )

    assert ctx.message is msg
    assert ctx.tool_results == []
    assert ctx.state is state
    assert ctx.new_messages == [msg]
