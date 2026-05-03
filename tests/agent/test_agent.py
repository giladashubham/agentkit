from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import pytest

from agentkit.agent import Agent, AgentConfig, AgentEvent, AgentEventType, AgentTool
from agentkit.llm import (
    AssistantMessage,
    Context,
    Message,
    Model,
    Response,
    RunOptions,
    StopReason,
    StreamEvent,
    StreamResponse,
    ToolCall,
    Usage,
)


def _stream_for(message: AssistantMessage, *, delay: float = 0) -> StreamResponse:
    async def events() -> AsyncIterator[StreamEvent]:
        if delay:
            await asyncio.sleep(delay)
        yield StreamEvent.start("test")
        if message.text():
            yield StreamEvent.text_delta(message.text(), partial=message)
        for call in message.tool_calls():
            yield StreamEvent.toolcall_end(call, partial=message)
        yield StreamEvent.done(
            Response(message=message, stop_reason=StopReason.STOP, usage=Usage(), model="test")
        )

    return StreamResponse(events())


async def test_run_awaits_completion_and_wait_for_idle_observes_active_run(
    mock_model: Model,
) -> None:
    agent = Agent(
        mock_model, stream_fn=lambda _m, _c, _o: _stream_for(Message.assistant("ok"), delay=0.05)
    )

    run_task = asyncio.create_task(agent.run("hi"))
    await asyncio.sleep(0)

    assert agent.is_running is True

    started = time.perf_counter()
    await agent.wait_for_idle()
    assert time.perf_counter() - started >= 0.04

    await run_task
    assert agent.is_running is False
    assert agent.state is not None
    assert agent.state.messages[-1].text() == "ok"


async def test_run_while_running_raises(mock_model: Model) -> None:
    agent = Agent(
        mock_model, stream_fn=lambda _m, _c, _o: _stream_for(Message.assistant("ok"), delay=0.05)
    )

    run_task = asyncio.create_task(agent.run("hi"))
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="already running"):
        await agent.run("again")
    await run_task


async def test_cancel_then_wait_for_idle_terminates_cleanly(mock_model: Model) -> None:
    agent = Agent(
        mock_model, stream_fn=lambda _m, _c, _o: _stream_for(Message.assistant("ok"), delay=0.05)
    )

    run_task = asyncio.create_task(agent.run("hi"))
    await asyncio.sleep(0)
    await agent.cancel()
    await run_task

    assert agent.is_running is False
    assert agent.state is not None
    assert agent.state.error_message == "cancelled"


async def test_subscribe_receives_events_and_unsubscribe_removes_listener(
    mock_model: Model,
) -> None:
    agent = Agent(mock_model, stream_fn=lambda _m, _c, _o: _stream_for(Message.assistant("ok")))
    events: list[AgentEventType] = []
    unsubscribe = agent.subscribe(lambda event: events.append(event.type))

    await agent.run("hi")
    unsubscribe()
    await agent.run("hi")

    assert events[0] == AgentEventType.AGENT_START
    assert events[-1] == AgentEventType.AGENT_END
    assert events.count(AgentEventType.AGENT_START) == 1


async def test_listener_exceptions_do_not_crash_loop(mock_model: Model) -> None:
    agent = Agent(mock_model, stream_fn=lambda _m, _c, _o: _stream_for(Message.assistant("ok")))
    events: list[AgentEventType] = []

    def broken_listener(_event: AgentEvent) -> None:
        raise RuntimeError("listener boom")

    agent.subscribe(broken_listener)
    agent.subscribe(lambda event: events.append(event.type))

    await agent.run("hi")

    assert AgentEventType.AGENT_END in events
    assert agent.state is not None
    assert agent.state.error_message is None


async def test_async_listener_is_awaited_before_next_event(mock_model: Model) -> None:
    agent = Agent(mock_model, stream_fn=lambda _m, _c, _o: _stream_for(Message.assistant("ok")))
    order: list[str] = []

    async def listener(event: AgentEvent) -> None:
        order.append(f"start:{event.type}")
        await asyncio.sleep(0)
        order.append(f"end:{event.type}")

    agent.subscribe(listener)

    await agent.run("hi")

    assert order[1].startswith("end:")
    assert order[2].startswith("start:")


async def test_steer_injects_message_after_current_turn(
    mock_model: Model, echo_agent_tool: AgentTool
) -> None:
    contexts: list[list[Message]] = []
    tool_call = ToolCall(id="tc", name="echo", arguments={"value": "hi"})

    def stream_fn(_model: Model, context: Context, _options: RunOptions | None) -> StreamResponse:
        contexts.append(context.messages)
        if len(contexts) == 1:
            return _stream_for(Message.assistant_with_tool_calls([tool_call]), delay=0.05)
        return _stream_for(Message.assistant("final"))

    agent = Agent(mock_model, tools=[echo_agent_tool], stream_fn=stream_fn)

    run_task = asyncio.create_task(agent.run("hi"))
    await asyncio.sleep(0.02)
    await agent.steer("steer")
    await run_task

    assert contexts[1][-1].text() == "steer"


async def test_follow_up_fires_when_agent_would_stop(mock_model: Model) -> None:
    contexts: list[list[Message]] = []

    def stream_fn(_model: Model, context: Context, _options: RunOptions | None) -> StreamResponse:
        contexts.append(context.messages)
        return _stream_for(Message.assistant("ok"), delay=0.03)

    agent = Agent(mock_model, stream_fn=stream_fn)

    run_task = asyncio.create_task(agent.run("hi"))
    await asyncio.sleep(0.01)
    await agent.follow_up("again")
    await run_task

    assert contexts[1][-1].text() == "again"


async def test_follow_up_not_fired_when_tool_calls_in_turn(
    mock_model: Model, echo_agent_tool: AgentTool
) -> None:
    contexts: list[list[Message]] = []
    tool_call = ToolCall(id="tc", name="echo", arguments={"value": "hi"})

    def stream_fn(_model: Model, context: Context, _options: RunOptions | None) -> StreamResponse:
        contexts.append(context.messages)
        if len(contexts) == 1:
            return _stream_for(Message.assistant_with_tool_calls([tool_call]), delay=0.03)
        return _stream_for(Message.assistant("final"))

    agent = Agent(mock_model, tools=[echo_agent_tool], stream_fn=stream_fn)

    run_task = asyncio.create_task(agent.run("hi"))
    await asyncio.sleep(0.01)
    await agent.follow_up("again")
    await run_task

    assert all(message.text() != "again" for message in contexts[1])


async def test_state_exists_before_first_run_and_updates(mock_model: Model) -> None:
    agent = Agent(mock_model, stream_fn=lambda _m, _c, _o: _stream_for(Message.assistant("ok")))

    assert agent.state.messages == []
    assert agent.state.is_streaming is False
    await agent.run("hi")

    assert agent.state.turn_index == 1


async def test_full_multi_turn_message_history(
    mock_model: Model, echo_agent_tool: AgentTool
) -> None:
    tool_call = ToolCall(id="tc", name="echo", arguments={"value": "hi"})
    calls = 0

    def stream_fn(_model: Model, _context: Context, _options: RunOptions | None) -> StreamResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _stream_for(Message.assistant_with_tool_calls([tool_call]))
        return _stream_for(Message.assistant("final"))

    agent = Agent(mock_model, tools=[echo_agent_tool], stream_fn=stream_fn)

    await agent.run("hi")

    assert agent.state is not None
    assert [message.role.value for message in agent.state.messages] == [
        "user",
        "assistant",
        "toolResult",
        "assistant",
    ]


async def test_agent_reuse_after_idle_preserves_history(mock_model: Model) -> None:
    contexts: list[list[Message]] = []

    def stream_fn(_model: Model, context: Context, _options: RunOptions | None) -> StreamResponse:
        contexts.append(context.messages)
        return _stream_for(Message.assistant("ok"))

    agent = Agent(mock_model, stream_fn=stream_fn)

    await agent.run("one")
    await agent.run("two")

    assert agent.state is not None
    assert [message.text() for message in agent.state.messages] == ["one", "ok", "two", "ok"]
    assert [message.text() for message in contexts[1]] == ["one", "ok", "two"]


async def test_per_run_system_prompt_and_tools_overrides(
    mock_model: Model, echo_agent_tool: AgentTool
) -> None:
    seen_contexts: list[Context] = []

    def stream_fn(_model: Model, context: Context, _options: RunOptions | None) -> StreamResponse:
        seen_contexts.append(context)
        return _stream_for(Message.assistant("ok"))

    agent = Agent(mock_model, system_prompt="base", stream_fn=stream_fn)

    await agent.run("hi", system_prompt="override", tools=[echo_agent_tool], config=AgentConfig())

    assert seen_contexts[0].system_prompt == "override"
    assert [tool.name for tool in seen_contexts[0].tools] == ["echo"]


async def test_reset_cancels_and_clears_state(mock_model: Model) -> None:
    agent = Agent(
        mock_model, stream_fn=lambda _m, _c, _o: _stream_for(Message.assistant("ok"), delay=0.1)
    )

    run_task = asyncio.create_task(agent.run("hi"))
    await asyncio.sleep(0)
    assert agent.is_running
    await agent.reset()
    await run_task

    assert not agent.is_running
    assert agent.state.messages == []
    assert agent.state.turn_index == 0
    assert agent.state.pending_tool_calls == []


async def test_clear_steering_queue_removes_queued_steer(mock_model: Model) -> None:
    agent = Agent(mock_model, stream_fn=lambda _m, _c, _o: _stream_for(Message.assistant("ok")))

    await agent.steer("steer1")
    await agent.steer("steer2")
    agent.clear_steering_queue()

    assert not agent.has_queued_messages()


async def test_clear_follow_up_queue(mock_model: Model) -> None:
    agent = Agent(mock_model, stream_fn=lambda _m, _c, _o: _stream_for(Message.assistant("ok")))

    await agent.follow_up("more")
    agent.clear_follow_up_queue()

    assert not agent.has_queued_messages()


async def test_clear_all_queues(mock_model: Model) -> None:
    agent = Agent(mock_model, stream_fn=lambda _m, _c, _o: _stream_for(Message.assistant("ok")))

    await agent.steer("s")
    await agent.follow_up("f")
    assert agent.has_queued_messages()
    agent.clear_all_queues()
    assert not agent.has_queued_messages()


async def test_has_queued_messages_returns_false_when_empty(mock_model: Model) -> None:
    agent = Agent(mock_model, stream_fn=lambda _m, _c, _o: _stream_for(Message.assistant("ok")))

    assert not agent.has_queued_messages()
    await agent.steer("msg")
    assert agent.has_queued_messages()


async def test_steering_mode_one_at_a_time(mock_model: Model) -> None:
    contexts: list[list[Message]] = []

    def stream_fn(_model: Model, context: Context, _options: RunOptions | None) -> StreamResponse:
        contexts.append(context.messages)
        return _stream_for(Message.assistant("ok"), delay=0.03)

    agent = Agent(
        mock_model,
        stream_fn=stream_fn,
        config=AgentConfig(steering_mode="one-at-a-time"),
    )

    run_task = asyncio.create_task(agent.run("hi"))
    await asyncio.sleep(0.01)
    await agent.steer("steer1")
    await agent.steer("steer2")
    await run_task

    assert not agent.has_queued_messages()
    assert contexts[1][-1].text() == "steer1"
    assert contexts[2][-1].text() == "steer2"


async def test_continue_resumes_from_user_or_tool_result_context(mock_model: Model) -> None:
    contexts: list[list[Message]] = []

    def stream_fn(_model: Model, context: Context, _options: RunOptions | None) -> StreamResponse:
        contexts.append(context.messages)
        return _stream_for(Message.assistant("ok"))

    agent = Agent(mock_model, stream_fn=stream_fn)

    await agent.run([Message.user("first")])
    assert agent.state is not None
    agent.state.messages = [Message.user("retry")]

    await agent.continue_()

    assert [message.text() for message in contexts[-1]] == ["retry"]
    assert [message.text() for message in agent.state.messages] == ["retry", "ok"]


async def test_continue_from_assistant_uses_queued_steering(mock_model: Model) -> None:
    contexts: list[list[Message]] = []

    def stream_fn(_model: Model, context: Context, _options: RunOptions | None) -> StreamResponse:
        contexts.append(context.messages)
        return _stream_for(Message.assistant("ok"))

    agent = Agent(mock_model, stream_fn=stream_fn)
    await agent.run("first")
    await agent.steer("steer")

    await agent.continue_()

    assert [message.text() for message in contexts[-1]] == ["first", "ok", "steer"]


async def test_continue_from_assistant_without_queue_raises(mock_model: Model) -> None:
    agent = Agent(mock_model, stream_fn=lambda _m, _c, _o: _stream_for(Message.assistant("ok")))
    await agent.run("first")

    with pytest.raises(RuntimeError, match="Cannot continue from message role: assistant"):
        await agent.continue_()


async def test_runtime_state_tracks_tools_and_full_run_streaming(
    mock_model: Model,
    echo_agent_tool: AgentTool,
) -> None:
    tool_call = ToolCall(id="tc", name="echo", arguments={"value": "hi"})
    calls = 0

    def stream_fn(_model: Model, _context: Context, _options: RunOptions | None) -> StreamResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _stream_for(Message.assistant_with_tool_calls([tool_call]))
        return _stream_for(Message.assistant("done"))

    agent = Agent(mock_model, tools=[echo_agent_tool], stream_fn=stream_fn)
    observed: list[tuple[AgentEventType, bool, list[str]]] = []

    def listener(event: AgentEvent) -> None:
        if event.type in {
            AgentEventType.TOOL_EXECUTION_START,
            AgentEventType.TOOL_EXECUTION_END,
            AgentEventType.AGENT_END,
        }:
            observed.append(
                (event.type, agent.state.is_streaming, list(agent.state.pending_tool_calls))
            )

    agent.subscribe(listener)

    await agent.run("hi")

    assert observed == [
        (AgentEventType.TOOL_EXECUTION_START, True, ["tc"]),
        (AgentEventType.TOOL_EXECUTION_END, True, []),
        (AgentEventType.AGENT_END, True, []),
    ]
    assert agent.state.is_streaming is False
