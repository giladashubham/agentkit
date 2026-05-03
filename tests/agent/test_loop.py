from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

from agentkit.agent import (
    AfterToolCallContext,
    AgentConfig,
    AgentEvent,
    AgentEventType,
    AgentState,
    AgentTool,
    AgentToolResult,
    BeforeToolCallContext,
    BeforeToolCallResult,
    run_agent_loop,
)
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
    tool,
)


def _stream_for(message: AssistantMessage) -> StreamResponse:
    async def events() -> AsyncIterator[StreamEvent]:
        yield StreamEvent.start("test")
        if message.text():
            yield StreamEvent.text_delta(message.text(), partial=message)
        for call in message.tool_calls():
            yield StreamEvent.toolcall_end(call, partial=message)
        yield StreamEvent.done(
            Response(message=message, stop_reason=StopReason.STOP, usage=Usage(), model="test")
        )

    return StreamResponse(events())


async def test_text_only_response_emits_events(mock_model: Model) -> None:
    events: list[AgentEvent] = []
    state = AgentState(mock_model)

    await run_agent_loop(
        [Message.user("hi")],
        state,
        AgentConfig(),
        events.append,
        stream_fn=lambda _m, _c, _o: _stream_for(Message.assistant("hello")),
    )

    assert [event.type for event in events] == [
        AgentEventType.AGENT_START,
        AgentEventType.TURN_START,
        AgentEventType.MESSAGE_START,  # initial user message
        AgentEventType.MESSAGE_END,
        AgentEventType.MESSAGE_START,  # assistant streaming
        AgentEventType.MESSAGE_UPDATE,
        AgentEventType.MESSAGE_END,
        AgentEventType.TURN_END,
        AgentEventType.AGENT_END,
    ]
    assert state.messages[-1].text() == "hello"


async def test_tool_call_executes_and_appends_tool_result(
    mock_model: Model, echo_agent_tool: AgentTool
) -> None:
    tool_call = ToolCall(id="tc", name="echo", arguments={"value": "hi"})
    events: list[AgentEvent] = []
    state = AgentState(mock_model, _tools=[echo_agent_tool])

    await run_agent_loop(
        [Message.user("hi")],
        state,
        AgentConfig(max_turns=1),
        events.append,
        stream_fn=lambda _m, _c, _o: _stream_for(Message.assistant_with_tool_calls([tool_call])),
    )

    assert AgentEventType.TOOL_EXECUTION_START in [event.type for event in events]
    assert AgentEventType.TOOL_EXECUTION_END in [event.type for event in events]
    assert state.messages[-1].content[0].content == "echo: hi"


async def test_terminate_tool_result_exits_after_one_tool_turn(mock_model: Model) -> None:
    @tool()
    def finish() -> str:
        return "done"

    class FinishTool(AgentTool):
        async def execute(self, *args, **kwargs) -> AgentToolResult:  # type: ignore[no-untyped-def]
            return AgentToolResult("done", terminate=True)

    tool_call = ToolCall(id="tc", name="finish", arguments={})
    state = AgentState(mock_model, _tools=[FinishTool(finish)])
    calls = 0

    def stream_fn(_model: Model, _context: Context, _options: RunOptions | None) -> StreamResponse:
        nonlocal calls
        calls += 1
        return _stream_for(Message.assistant_with_tool_calls([tool_call]))

    await run_agent_loop(
        [Message.user("hi")], state, AgentConfig(), lambda _e: None, stream_fn=stream_fn
    )

    assert calls == 1
    assert state.turn_index == 1


async def test_terminate_requires_all_tools_to_agree(mock_model: Model) -> None:
    """Only terminate when ALL tools in a batch return terminate=True."""

    @tool()
    def stop_tool() -> str:
        return "stop"

    @tool()
    def continue_tool() -> str:
        return "continue"

    class StopTool(AgentTool):
        async def execute(self, *args, **kwargs) -> AgentToolResult:  # type: ignore[no-untyped-def]
            return AgentToolResult("stop", terminate=True)

    class ContinueTool(AgentTool):
        async def execute(self, *args, **kwargs) -> AgentToolResult:  # type: ignore[no-untyped-def]
            return AgentToolResult("continue", terminate=False)

    calls = [
        ToolCall(id="1", name="stop_tool", arguments={}),
        ToolCall(id="2", name="continue_tool", arguments={}),
    ]
    state = AgentState(mock_model, _tools=[StopTool(stop_tool), ContinueTool(continue_tool)])
    llm_calls = 0

    def stream_fn(_model: Model, _context: Context, _options: RunOptions | None) -> StreamResponse:
        nonlocal llm_calls
        llm_calls += 1
        if llm_calls == 1:
            return _stream_for(Message.assistant_with_tool_calls(calls))
        return _stream_for(Message.assistant("final"))

    await run_agent_loop(
        [Message.user("hi")], state, AgentConfig(), lambda _e: None, stream_fn=stream_fn
    )

    # Should NOT terminate — only one of two tools said terminate=True
    assert llm_calls == 2


async def test_max_turns_exits_after_first_turn_even_with_tool_calls(
    mock_model: Model, echo_agent_tool: AgentTool
) -> None:
    tool_call = ToolCall(id="tc", name="echo", arguments={"value": "hi"})
    state = AgentState(mock_model, _tools=[echo_agent_tool])
    calls = 0

    def stream_fn(_model: Model, _context: Context, _options: RunOptions | None) -> StreamResponse:
        nonlocal calls
        calls += 1
        return _stream_for(Message.assistant_with_tool_calls([tool_call]))

    await run_agent_loop(
        [Message.user("hi")], state, AgentConfig(max_turns=1), lambda _e: None, stream_fn=stream_fn
    )

    assert calls == 1


async def test_cancellation_signal_exits_before_turn(mock_model: Model) -> None:
    signal = asyncio.Event()
    signal.set()
    events: list[AgentEvent] = []
    state = AgentState(mock_model)

    await run_agent_loop(
        [Message.user("hi")],
        state,
        AgentConfig(),
        events.append,
        signal=signal,
        stream_fn=lambda _m, _c, _o: _stream_for(Message.assistant("never")),
    )

    assert [event.type for event in events] == [
        AgentEventType.AGENT_START,
        AgentEventType.TURN_START,
        AgentEventType.MESSAGE_START,
        AgentEventType.MESSAGE_END,
        AgentEventType.AGENT_END,
    ]
    assert state.error_message == "cancelled"


async def test_unknown_tool_name_returns_error_result(mock_model: Model) -> None:
    tool_call = ToolCall(id="tc", name="missing", arguments={})
    state = AgentState(mock_model)

    await run_agent_loop(
        [Message.user("hi")],
        state,
        AgentConfig(max_turns=1),
        lambda _e: None,
        stream_fn=lambda _m, _c, _o: _stream_for(Message.assistant_with_tool_calls([tool_call])),
    )

    assert "Unknown tool: missing" in state.messages[-1].content[0].content


async def test_transform_context_messages_passed_to_stream_fn(mock_model: Model) -> None:
    seen: list[Message] = []

    def transform(_messages: list[Message]) -> list[Message]:
        return [Message.user("transformed")]

    def stream_fn(_model: Model, context: Context, _options: RunOptions | None) -> StreamResponse:
        seen.extend(context.messages)
        return _stream_for(Message.assistant("ok"))

    await run_agent_loop(
        [Message.user("original")],
        AgentState(mock_model),
        AgentConfig(transform_context=transform),
        lambda _e: None,
        stream_fn=stream_fn,
    )

    assert [message.text() for message in seen] == ["transformed"]


async def test_parallel_tools_are_concurrent(mock_model: Model) -> None:
    class SleepTool(AgentTool):
        async def execute(self, *args, **kwargs) -> AgentToolResult:  # type: ignore[no-untyped-def]
            await asyncio.sleep(0.05)
            return AgentToolResult("ok")

    @tool()
    def first() -> str:
        return "unused"

    @tool()
    def second() -> str:
        return "unused"

    calls = [
        ToolCall(id="1", name="first", arguments={}),
        ToolCall(id="2", name="second", arguments={}),
    ]
    state = AgentState(mock_model, _tools=[SleepTool(first), SleepTool(second)])

    started = time.perf_counter()
    await run_agent_loop(
        [Message.user("hi")],
        state,
        AgentConfig(max_turns=1),
        lambda _e: None,
        stream_fn=lambda _m, _c, _o: _stream_for(Message.assistant_with_tool_calls(calls)),
    )

    assert time.perf_counter() - started < 0.09


async def test_sequential_tools_run_in_order(mock_model: Model) -> None:
    order: list[str] = []

    class RecordingTool(AgentTool):
        async def execute(self, *args, **kwargs) -> AgentToolResult:  # type: ignore[no-untyped-def]
            order.append(self.name)
            return AgentToolResult("ok")

    @tool()
    def first() -> str:
        return "unused"

    @tool()
    def second() -> str:
        return "unused"

    calls = [
        ToolCall(id="1", name="first", arguments={}),
        ToolCall(id="2", name="second", arguments={}),
    ]
    state = AgentState(
        mock_model,
        _tools=[
            RecordingTool(first, execution_mode="sequential"),
            RecordingTool(second, execution_mode="sequential"),
        ],
    )

    await run_agent_loop(
        [Message.user("hi")],
        state,
        AgentConfig(max_turns=1),
        lambda _e: None,
        stream_fn=lambda _m, _c, _o: _stream_for(Message.assistant_with_tool_calls(calls)),
    )

    assert order == ["first", "second"]


async def test_mixed_parallel_and_sequential_forces_entire_batch_sequential(
    mock_model: Model,
) -> None:
    order: list[str] = []

    class NameTool(AgentTool):
        async def execute(self, *args, **kwargs) -> AgentToolResult:  # type: ignore[no-untyped-def]
            order.append(self.name)
            return AgentToolResult(self.name)

    @tool()
    def first() -> str:
        return "unused"

    @tool()
    def second() -> str:
        return "unused"

    @tool()
    def third() -> str:
        return "unused"

    calls = [
        ToolCall(id="1", name="first", arguments={}),
        ToolCall(id="2", name="second", arguments={}),
        ToolCall(id="3", name="third", arguments={}),
    ]
    state = AgentState(
        mock_model,
        _tools=[
            NameTool(first, execution_mode="parallel"),
            NameTool(second, execution_mode="sequential"),
            NameTool(third, execution_mode="parallel"),
        ],
    )

    await run_agent_loop(
        [Message.user("hi")],
        state,
        AgentConfig(max_turns=1),
        lambda _e: None,
        stream_fn=lambda _m, _c, _o: _stream_for(Message.assistant_with_tool_calls(calls)),
    )

    result_messages = state.messages[-3:]
    assert order == ["first", "second", "third"]
    assert [message.content[0].content for message in result_messages] == [
        "first",
        "second",
        "third",
    ]


async def test_tool_update_tasks_settle_before_error_end_event(mock_model: Model) -> None:
    class UpdatingFailingTool(AgentTool):
        async def execute(self, *args, **kwargs) -> AgentToolResult:  # type: ignore[no-untyped-def]
            on_update = args[3]
            on_update("progress")
            raise RuntimeError("boom")

    @tool()
    def failing() -> str:
        return "unused"

    tool_call = ToolCall(id="tc", name="failing", arguments={})
    event_order: list[AgentEventType] = []

    async def emit(event: AgentEvent) -> None:
        if event.type == AgentEventType.TOOL_EXECUTION_UPDATE:
            await asyncio.sleep(0.01)
        event_order.append(event.type)

    await run_agent_loop(
        [Message.user("hi")],
        AgentState(mock_model, _tools=[UpdatingFailingTool(failing)]),
        AgentConfig(max_turns=1),
        emit,
        stream_fn=lambda _m, _c, _o: _stream_for(Message.assistant_with_tool_calls([tool_call])),
    )
    await asyncio.sleep(0.02)

    update_index = event_order.index(AgentEventType.TOOL_EXECUTION_UPDATE)
    end_index = event_order.index(AgentEventType.TOOL_EXECUTION_END)
    assert update_index < end_index


async def test_before_tool_hook_receives_context(
    mock_model: Model, echo_agent_tool: AgentTool
) -> None:
    contexts: list[BeforeToolCallContext] = []
    tool_call = ToolCall(id="tc", name="echo", arguments={"value": "hi"})

    async def before(ctx: BeforeToolCallContext) -> None:
        contexts.append(ctx)

    await run_agent_loop(
        [Message.user("hi")],
        AgentState(mock_model, _tools=[echo_agent_tool]),
        AgentConfig(max_turns=1, before_tool_call=before),
        lambda _e: None,
        stream_fn=lambda _m, _c, _o: _stream_for(Message.assistant_with_tool_calls([tool_call])),
    )

    assert len(contexts) == 1
    assert contexts[0].tool_call.name == "echo"
    assert contexts[0].agent_tool is echo_agent_tool
    assert contexts[0].arguments == {"value": "hi"}
    assert contexts[0].args == {"value": "hi"}


async def test_after_tool_hook_receives_context(
    mock_model: Model, echo_agent_tool: AgentTool
) -> None:
    contexts: list[AfterToolCallContext] = []
    tool_call = ToolCall(id="tc", name="echo", arguments={"value": "hi"})

    def after(ctx: AfterToolCallContext) -> None:
        contexts.append(ctx)

    await run_agent_loop(
        [Message.user("hi")],
        AgentState(mock_model, _tools=[echo_agent_tool]),
        AgentConfig(max_turns=1, after_tool_call=after),
        lambda _e: None,
        stream_fn=lambda _m, _c, _o: _stream_for(Message.assistant_with_tool_calls([tool_call])),
    )

    assert len(contexts) == 1
    assert contexts[0].tool_call.name == "echo"
    assert contexts[0].result.content == "echo: hi"
    assert contexts[0].is_error is False
    assert contexts[0].arguments == {"value": "hi"}
    assert contexts[0].args == {"value": "hi"}


async def test_before_tool_hook_can_block_execution(
    mock_model: Model, echo_agent_tool: AgentTool
) -> None:
    tool_call = ToolCall(id="tc", name="echo", arguments={"value": "hi"})
    executed = False

    class TrackingTool(AgentTool):
        async def execute(self, *args, **kwargs) -> AgentToolResult:  # type: ignore[no-untyped-def]
            nonlocal executed
            executed = True
            return AgentToolResult("should not reach")

    @tool()
    def echo(value: str) -> str:
        return f"echo: {value}"

    def before(_ctx: BeforeToolCallContext) -> BeforeToolCallResult:
        return BeforeToolCallResult("blocked by hook", is_error=True)

    state = AgentState(mock_model, _tools=[TrackingTool(echo_agent_tool.tool)])
    await run_agent_loop(
        [Message.user("hi")],
        state,
        AgentConfig(max_turns=1, before_tool_call=before),
        lambda _e: None,
        stream_fn=lambda _m, _c, _o: _stream_for(Message.assistant_with_tool_calls([tool_call])),
    )

    assert not executed
    assert state_last_tool_result(state) == "blocked by hook"
    assert state.messages[-1].content[0].is_error is True


async def test_after_tool_hook_can_override_result(
    mock_model: Model, echo_agent_tool: AgentTool
) -> None:
    tool_call = ToolCall(id="tc", name="echo", arguments={"value": "hi"})

    def after(_ctx: AfterToolCallContext) -> AgentToolResult:
        return AgentToolResult("overridden")

    state = AgentState(mock_model, _tools=[echo_agent_tool])
    await run_agent_loop(
        [Message.user("hi")],
        state,
        AgentConfig(max_turns=1, after_tool_call=after),
        lambda _e: None,
        stream_fn=lambda _m, _c, _o: _stream_for(Message.assistant_with_tool_calls([tool_call])),
    )

    assert state.messages[-1].content[0].content == "overridden"


def state_last_tool_result(state: AgentState) -> str:
    if not state.messages:
        return ""
    last = state.messages[-1]
    if hasattr(last, "content") and last.content:
        item = last.content[0]
        return item.content if hasattr(item, "content") else str(item)
    return ""


async def test_should_stop_after_turn_hook(mock_model: Model) -> None:
    calls = 0

    def stream_fn(_model: Model, _context: Context, _options: RunOptions | None) -> StreamResponse:
        nonlocal calls
        calls += 1
        return _stream_for(Message.assistant("ok"))

    stop_after = 0

    seen_new_message_counts: list[int] = []

    def should_stop(ctx) -> bool:  # type: ignore[no-untyped-def]
        nonlocal stop_after
        stop_after += 1
        seen_new_message_counts.append(len(ctx.new_messages))
        assert ctx.message.role.value == "assistant"
        return stop_after >= 2

    await run_agent_loop(
        [Message.user("hi")],
        AgentState(mock_model),
        AgentConfig(
            _get_follow_up_messages=lambda: [Message.user("more")] if stop_after < 2 else [],
            should_stop_after_turn=should_stop,
        ),
        lambda _e: None,
        stream_fn=stream_fn,
    )

    assert calls == 2
    assert seen_new_message_counts == [2, 4]


async def test_steer_messages_polled_before_first_llm_call(mock_model: Model) -> None:
    """Steer queue is checked at the top of each iteration, including turn 0."""
    contexts: list[list[Message]] = []
    steer_injected = False

    def get_steer() -> list[Message]:
        nonlocal steer_injected
        if not steer_injected:
            steer_injected = True
            return [Message.user("pre-steer")]
        return []

    def stream_fn(_model: Model, context: Context, _options: RunOptions | None) -> StreamResponse:
        contexts.append(context.messages)
        return _stream_for(Message.assistant("ok"))

    await run_agent_loop(
        [Message.user("hi")],
        AgentState(mock_model),
        AgentConfig(_get_steer_messages=get_steer),
        lambda _e: None,
        stream_fn=stream_fn,
    )

    # steer message should be in the first LLM call's context
    assert any(m.text() == "pre-steer" for m in contexts[0])


async def test_steer_messages_are_injected_after_tool_turn(
    mock_model: Model, echo_agent_tool: AgentTool
) -> None:
    """Steer messages queued after context capture appear before the NEXT LLM call."""
    contexts: list[list[Message]] = []
    tool_call = ToolCall(id="tc", name="echo", arguments={"value": "hi"})

    def get_steer() -> list[Message]:
        # Return steer only after the first LLM call has already happened
        if len(contexts) == 0:
            return []
        if len(contexts) == 1:
            return [Message.user("steer")]
        return []

    def stream_fn(_model: Model, context: Context, _options: RunOptions | None) -> StreamResponse:
        contexts.append(context.messages)
        if len(contexts) == 1:
            return _stream_for(Message.assistant_with_tool_calls([tool_call]))
        return _stream_for(Message.assistant("final"))

    await run_agent_loop(
        [Message.user("hi")],
        AgentState(mock_model, _tools=[echo_agent_tool]),
        AgentConfig(_get_steer_messages=get_steer),
        lambda _e: None,
        stream_fn=stream_fn,
    )

    assert contexts[1][-1].text() == "steer"


async def test_follow_up_messages_are_injected_only_at_natural_stop(mock_model: Model) -> None:
    contexts: list[list[Message]] = []
    follow_up_used = False

    def get_follow_up() -> list[Message]:
        nonlocal follow_up_used
        if follow_up_used:
            return []
        follow_up_used = True
        return [Message.user("again")]

    def stream_fn(_model: Model, context: Context, _options: RunOptions | None) -> StreamResponse:
        contexts.append(context.messages)
        return _stream_for(Message.assistant("ok"))

    await run_agent_loop(
        [Message.user("hi")],
        AgentState(mock_model),
        AgentConfig(_get_follow_up_messages=get_follow_up),
        lambda _e: None,
        stream_fn=stream_fn,
    )

    assert len(contexts) == 2
    assert contexts[1][-1].text() == "again"


async def test_tool_execution_end_includes_is_error_flag(
    mock_model: Model, echo_agent_tool: AgentTool
) -> None:
    tool_call = ToolCall(id="tc", name="echo", arguments={"value": "hi"})
    end_events: list[AgentEvent] = []

    def emit(event: AgentEvent) -> None:
        if event.type == AgentEventType.TOOL_EXECUTION_END:
            end_events.append(event)

    await run_agent_loop(
        [Message.user("hi")],
        AgentState(mock_model, _tools=[echo_agent_tool]),
        AgentConfig(max_turns=1),
        emit,
        stream_fn=lambda _m, _c, _o: _stream_for(Message.assistant_with_tool_calls([tool_call])),
    )

    assert len(end_events) == 1
    _tc, _result, is_error = end_events[0].data
    assert is_error is False


async def test_agent_end_carries_new_messages(mock_model: Model) -> None:
    end_events: list[AgentEvent] = []

    def emit(event: AgentEvent) -> None:
        if event.type == AgentEventType.AGENT_END:
            end_events.append(event)

    await run_agent_loop(
        [Message.user("hi")],
        AgentState(mock_model),
        AgentConfig(),
        emit,
        stream_fn=lambda _m, _c, _o: _stream_for(Message.assistant("hello")),
    )

    assert len(end_events) == 1
    new_messages = end_events[0].data
    assert len(new_messages) == 2  # user + assistant
    assert new_messages[0].text() == "hi"
    assert new_messages[1].text() == "hello"


async def test_turn_end_carries_assistant_message_and_tool_results(
    mock_model: Model, echo_agent_tool: AgentTool
) -> None:
    tool_call = ToolCall(id="tc", name="echo", arguments={"value": "test"})
    turn_end_events: list[AgentEvent] = []

    def emit(event: AgentEvent) -> None:
        if event.type == AgentEventType.TURN_END:
            turn_end_events.append(event)

    await run_agent_loop(
        [Message.user("hi")],
        AgentState(mock_model, _tools=[echo_agent_tool]),
        AgentConfig(max_turns=1),
        emit,
        stream_fn=lambda _m, _c, _o: _stream_for(Message.assistant_with_tool_calls([tool_call])),
    )

    assert len(turn_end_events) == 1
    turn_idx, assistant_msg, tool_results = turn_end_events[0].data
    assert turn_idx == 0
    assert len(assistant_msg.tool_calls()) == 1
    assert len(tool_results) == 1


async def test_message_update_carries_raw_stream_event(mock_model: Model) -> None:
    update_events: list[AgentEvent] = []

    def emit(event: AgentEvent) -> None:
        if event.type == AgentEventType.MESSAGE_UPDATE:
            update_events.append(event)

    await run_agent_loop(
        [Message.user("hi")],
        AgentState(mock_model),
        AgentConfig(),
        emit,
        stream_fn=lambda _m, _c, _o: _stream_for(Message.assistant("hi")),
    )

    assert len(update_events) >= 1
    msg, raw_event = update_events[0].data
    assert isinstance(msg, AssistantMessage)
    assert raw_event is not None


async def test_before_tool_hook_sees_prepared_validated_arguments(mock_model: Model) -> None:
    @tool()
    def echo(value: str) -> str:
        return f"echo: {value}"

    seen_arguments: list[dict] = []
    executed_arguments: list[str] = []

    class RecordingTool(AgentTool):
        async def execute_validated(self, tool_call_id, arguments, signal=None, on_update=None):  # type: ignore[no-untyped-def]
            del tool_call_id, signal, on_update
            executed_arguments.append(arguments["value"])
            return AgentToolResult(f"echo: {arguments['value']}")

    def prepare(arguments: dict) -> dict:
        return {"value": f"value-{arguments['value']}"}

    def before(ctx: BeforeToolCallContext) -> None:
        seen_arguments.append(dict(ctx.arguments))
        ctx.arguments["value"] = ctx.arguments["value"].upper()

    tool_call = ToolCall(id="tc", name="echo", arguments={"value": 123})
    state = AgentState(mock_model, _tools=[RecordingTool(echo, prepare_arguments=prepare)])

    await run_agent_loop(
        [Message.user("hi")],
        state,
        AgentConfig(max_turns=1, before_tool_call=before),
        lambda _e: None,
        stream_fn=lambda _m, _c, _o: _stream_for(Message.assistant_with_tool_calls([tool_call])),
    )

    assert seen_arguments == [{"value": "value-123"}]
    assert executed_arguments == ["VALUE-123"]


async def test_sequential_termination_requires_all_tools_to_agree(mock_model: Model) -> None:
    @tool()
    def stop_tool() -> str:
        return "stop"

    @tool()
    def continue_tool() -> str:
        return "continue"

    executed: list[str] = []

    class StopTool(AgentTool):
        async def execute(self, *args, **kwargs) -> AgentToolResult:  # type: ignore[no-untyped-def]
            executed.append(self.name)
            return AgentToolResult("stop", terminate=True)

    class ContinueTool(AgentTool):
        async def execute(self, *args, **kwargs) -> AgentToolResult:  # type: ignore[no-untyped-def]
            executed.append(self.name)
            return AgentToolResult("continue", terminate=False)

    calls = [
        ToolCall(id="1", name="stop_tool", arguments={}),
        ToolCall(id="2", name="continue_tool", arguments={}),
    ]
    state = AgentState(
        mock_model,
        _tools=[
            StopTool(stop_tool, execution_mode="sequential"),
            ContinueTool(continue_tool, execution_mode="sequential"),
        ],
    )
    llm_calls = 0

    def stream_fn(_model: Model, _context: Context, _options: RunOptions | None) -> StreamResponse:
        nonlocal llm_calls
        llm_calls += 1
        if llm_calls == 1:
            return _stream_for(Message.assistant_with_tool_calls(calls))
        return _stream_for(Message.assistant("final"))

    await run_agent_loop(
        [Message.user("hi")], state, AgentConfig(), lambda _e: None, stream_fn=stream_fn
    )

    assert executed == ["stop_tool", "continue_tool"]
    assert llm_calls == 2


async def test_stream_error_appends_assistant_error_and_turn_end(mock_model: Model) -> None:
    events: list[AgentEvent] = []

    def stream_fn(_model: Model, _context: Context, _options: RunOptions | None) -> StreamResponse:
        async def events() -> AsyncIterator[StreamEvent]:
            partial = Message.assistant("partial")
            yield StreamEvent.start("test", partial=partial)
            yield StreamEvent.error(RuntimeError("boom"), partial=partial)

        return StreamResponse(events())

    state = AgentState(mock_model)
    await run_agent_loop(
        [Message.user("hi")],
        state,
        AgentConfig(),
        events.append,
        stream_fn=stream_fn,
    )

    assert [event.type for event in events][-3:] == [
        AgentEventType.MESSAGE_END,
        AgentEventType.TURN_END,
        AgentEventType.AGENT_END,
    ]
    assert state.messages[-1].role.value == "assistant"
    assert state.messages[-1].text() == "partial"
    assert state.messages[-1].stop_reason == StopReason.ERROR.value
    assert state.messages[-1].error_message == "boom"


async def test_parallel_tool_preflight_finishes_before_executions(mock_model: Model) -> None:
    preflight_order: list[str] = []
    execution_preflight_counts: list[int] = []

    class RecordingTool(AgentTool):
        async def execute(self, *args, **kwargs) -> AgentToolResult:  # type: ignore[no-untyped-def]
            execution_preflight_counts.append(len(preflight_order))
            await asyncio.sleep(0.01)
            return AgentToolResult(self.name)

    @tool()
    def first() -> str:
        return "first"

    @tool()
    def second() -> str:
        return "second"

    def before(ctx: BeforeToolCallContext) -> None:
        preflight_order.append(ctx.tool_call.name)

    calls = [
        ToolCall(id="1", name="first", arguments={}),
        ToolCall(id="2", name="second", arguments={}),
    ]

    await run_agent_loop(
        [Message.user("hi")],
        AgentState(mock_model, _tools=[RecordingTool(first), RecordingTool(second)]),
        AgentConfig(max_turns=1, before_tool_call=before),
        lambda _e: None,
        stream_fn=lambda _m, _c, _o: _stream_for(Message.assistant_with_tool_calls(calls)),
    )

    assert preflight_order == ["first", "second"]
    assert execution_preflight_counts == [2, 2]


async def test_tool_hook_exceptions_become_tool_errors(
    mock_model: Model,
    echo_agent_tool: AgentTool,
) -> None:
    tool_call = ToolCall(id="tc", name="echo", arguments={"value": "hi"})

    def before(_ctx: BeforeToolCallContext) -> None:
        raise RuntimeError("before boom")

    state = AgentState(mock_model, _tools=[echo_agent_tool])
    await run_agent_loop(
        [Message.user("hi")],
        state,
        AgentConfig(max_turns=1, before_tool_call=before),
        lambda _e: None,
        stream_fn=lambda _m, _c, _o: _stream_for(Message.assistant_with_tool_calls([tool_call])),
    )

    assert state.messages[-1].content[0].content == "before boom"
    assert state.messages[-1].content[0].is_error is True

    def after(_ctx: AfterToolCallContext) -> AgentToolResult:
        raise RuntimeError("after boom")

    state = AgentState(mock_model, _tools=[echo_agent_tool])
    await run_agent_loop(
        [Message.user("hi")],
        state,
        AgentConfig(max_turns=1, after_tool_call=after),
        lambda _e: None,
        stream_fn=lambda _m, _c, _o: _stream_for(Message.assistant_with_tool_calls([tool_call])),
    )

    assert state.messages[-1].content[0].content == "after boom"
    assert state.messages[-1].content[0].is_error is True


async def test_after_tool_hook_can_override_is_error(
    mock_model: Model,
    echo_agent_tool: AgentTool,
) -> None:
    tool_call = ToolCall(id="tc", name="echo", arguments={"value": "hi"})

    def after(_ctx: AfterToolCallContext) -> AgentToolResult:
        return AgentToolResult("not actually an error", is_error=True)

    state = AgentState(mock_model, _tools=[echo_agent_tool])
    await run_agent_loop(
        [Message.user("hi")],
        state,
        AgentConfig(max_turns=1, after_tool_call=after),
        lambda _e: None,
        stream_fn=lambda _m, _c, _o: _stream_for(Message.assistant_with_tool_calls([tool_call])),
    )

    assert state.messages[-1].content[0].content == "not actually an error"
    assert state.messages[-1].content[0].is_error is True


async def test_unexpected_loop_failure_appends_assistant_error(mock_model: Model) -> None:
    def transform(_messages: list[Message]) -> list[Message]:
        raise RuntimeError("transform boom")

    state = AgentState(mock_model)
    await run_agent_loop(
        [Message.user("hi")],
        state,
        AgentConfig(transform_context=transform),
        lambda _e: None,
        stream_fn=lambda _m, _c, _o: _stream_for(Message.assistant("never")),
    )

    assert state.messages[-1].role.value == "assistant"
    assert state.messages[-1].stop_reason == StopReason.ERROR.value
    assert state.messages[-1].error_message == "transform boom"
