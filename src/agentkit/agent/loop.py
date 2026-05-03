from __future__ import annotations

import asyncio
import dataclasses
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from agentkit.exceptions import ToolValidationError
from agentkit.llm import stream as llm_stream
from agentkit.llm.context import Context
from agentkit.llm.model import RunOptions
from agentkit.llm.streaming import EventType, StreamResponse
from agentkit.llm.types import AssistantMessage, Message, StopReason, ToolCall

from .config import AgentConfig
from .state import AgentState
from .tool import AgentTool
from .types import (
    AfterToolCallContext,
    AgentEvent,
    AgentToolResult,
    BeforeToolCallContext,
    BeforeToolCallResult,
    ShouldStopAfterTurnContext,
    StreamFn,
)

__all__ = ["run_agent_loop"]

EmitFn = Callable[[AgentEvent], Awaitable[None] | None]


@dataclass(slots=True)
class _PreparedToolCall:
    index: int
    tool_call: ToolCall
    agent_tool: AgentTool
    arguments: dict[str, Any]
    assistant_message: AssistantMessage


@dataclass(slots=True)
class _ToolExecutionOutcome:
    index: int
    tool_call: ToolCall
    result: AgentToolResult
    is_error: bool = False


async def run_agent_loop(
    initial_messages: list[Message],
    state: AgentState,
    config: AgentConfig,
    emit: EmitFn,
    signal: asyncio.Event | None = None,
    stream_fn: StreamFn | None = None,
) -> None:
    """Run an agent loop until the model stops, max turns are reached, or cancelled."""
    stream_fn = stream_fn or llm_stream
    pre_run_count = len(state.messages)
    turns_this_run = 0
    pending_messages = list(initial_messages)
    state.is_streaming = True
    state.streaming_message = None
    state.pending_tool_calls = []

    await _emit(emit, AgentEvent.agent_start())

    try:
        pending_messages.extend(await _get_messages(config._get_steer_messages))

        while True:
            has_more_tool_calls = True

            while has_more_tool_calls or pending_messages:
                if config.max_turns is not None and turns_this_run >= config.max_turns:
                    return

                await _emit(emit, AgentEvent.turn_start(state.turn_index))

                if pending_messages:
                    for msg in pending_messages:
                        await _emit(emit, AgentEvent.message_start(msg))
                        await _emit(emit, AgentEvent.message_end(msg))
                    state.messages = [*state.messages, *pending_messages]
                    pending_messages = []

                if signal is not None and signal.is_set():
                    state.error_message = "cancelled"
                    return

                context = await _build_context(state, config)
                run_options = _build_run_options(config.run_options or state.run_options, signal)
                stream_response = stream_fn(state.model, context, run_options)
                assistant_msg = await _consume_stream(stream_response, state, emit, signal)
                state.messages = [*state.messages, assistant_msg]

                tool_result_messages: list[Message] = []
                tool_calls = assistant_msg.tool_calls()
                should_terminate = False

                if _is_error_or_aborted(assistant_msg):
                    await _emit(
                        emit,
                        AgentEvent.turn_end(state.turn_index, assistant_msg, tool_result_messages),
                    )
                    turns_this_run += 1
                    state.turn_index += 1
                    return

                if tool_calls:
                    tool_results, should_terminate = await _execute_tools(
                        tool_calls, state, config, emit, signal, assistant_msg
                    )
                    for outcome in tool_results:
                        trm = Message.tool_result(
                            outcome.tool_call.id,
                            outcome.tool_call.name,
                            outcome.result.content,
                            is_error=outcome.is_error,
                        )
                        tool_result_messages.append(trm)
                    for trm in tool_result_messages:
                        await _emit(emit, AgentEvent.message_start(trm))
                        await _emit(emit, AgentEvent.message_end(trm))
                    state.messages = [*state.messages, *tool_result_messages]

                await _emit(
                    emit,
                    AgentEvent.turn_end(state.turn_index, assistant_msg, tool_result_messages),
                )
                turns_this_run += 1
                state.turn_index += 1

                if signal is not None and signal.is_set():
                    state.error_message = "cancelled"
                    return

                stop_context = ShouldStopAfterTurnContext(
                    assistant_message=assistant_msg,
                    tool_result_messages=tool_result_messages,
                    state=state,
                    new_messages=state.messages[pre_run_count:],
                )
                if await _should_stop_after_turn(config, stop_context):
                    return

                if config.max_turns is not None and turns_this_run >= config.max_turns:
                    return

                pending_messages = await _get_messages(config._get_steer_messages)
                has_more_tool_calls = bool(tool_calls) and not should_terminate

            follow_up_messages = await _get_messages(config._get_follow_up_messages)
            if follow_up_messages:
                pending_messages = follow_up_messages
                continue

            break
    except asyncio.CancelledError:
        state.error_message = "cancelled"
    except Exception as exc:
        await _append_failure_message(exc, state, emit, signal)
    finally:
        if signal is not None and signal.is_set():
            state.error_message = "cancelled"
        state.streaming_message = None
        state.pending_tool_calls = []
        new_messages = state.messages[pre_run_count:]
        await _emit(emit, AgentEvent.agent_end(new_messages))
        state.is_streaming = False


async def _build_context(state: AgentState, config: AgentConfig) -> Context:
    messages_for_llm = state.messages
    if config.transform_context is not None:
        transformed = config.transform_context(messages_for_llm)
        messages_for_llm = await transformed if inspect.isawaitable(transformed) else transformed

    return Context(
        messages=messages_for_llm,
        system_prompt=state.system_prompt,
        tools=state.llm_tools(),
    )


async def _consume_stream(
    stream_response: StreamResponse,
    state: AgentState,
    emit: EmitFn,
    signal: asyncio.Event | None = None,
) -> AssistantMessage:
    state.streaming_message = AssistantMessage(content=[])
    await _emit(emit, AgentEvent.message_start(state.streaming_message))

    final_message: AssistantMessage | None = None
    try:
        async for event in stream_response:
            if signal is not None and signal.is_set():
                final_message = _error_assistant_message(
                    asyncio.CancelledError(StopReason.ABORTED.value),
                    state.streaming_message,
                    aborted=True,
                )
                state.error_message = final_message.error_message
                state.streaming_message = final_message
                await _emit(emit, AgentEvent.message_end(final_message))
                break

            if event.type == EventType.START:
                if isinstance(event.partial, AssistantMessage):
                    state.streaming_message = event.partial
                continue
            if event.type == EventType.ERROR:
                error = (
                    event.data
                    if isinstance(event.data, Exception)
                    else RuntimeError(str(event.data))
                )
                final_message = _error_assistant_message(error, event.partial, signal=signal)
                state.error_message = final_message.error_message
                state.streaming_message = final_message
                await _emit(emit, AgentEvent.message_end(final_message))
                break
            if event.type == EventType.DONE:
                if not isinstance(event.partial, AssistantMessage):
                    raise TypeError("stream DONE event did not contain an AssistantMessage")
                final_message = event.partial
                state.streaming_message = final_message
                await _emit(emit, AgentEvent.message_end(final_message))
                break

            if isinstance(event.partial, AssistantMessage):
                state.streaming_message = event.partial
                await _emit(emit, AgentEvent.message_update(event.partial, event))

        if final_message is None:
            if isinstance(state.streaming_message, AssistantMessage):
                final_message = state.streaming_message
            else:
                final_message = AssistantMessage(content=[])
            await _emit(emit, AgentEvent.message_end(final_message))
        return final_message
    finally:
        state.streaming_message = None


async def _execute_tools(
    tool_calls: list[ToolCall],
    state: AgentState,
    config: AgentConfig,
    emit: EmitFn,
    signal: asyncio.Event | None = None,
    assistant_msg: AssistantMessage | None = None,
) -> tuple[list[_ToolExecutionOutcome], bool]:
    force_sequential = config.tool_execution == "sequential" or any(
        _execution_mode(call, state.tool_lookup()) == "sequential" for call in tool_calls
    )

    prepared_entries: list[_PreparedToolCall | _ToolExecutionOutcome] = []
    for index, call in enumerate(tool_calls):
        prepared_entries.append(
            await _prepare_tool_call(index, call, state, config, emit, signal, assistant_msg)
        )

    if force_sequential:
        results: list[_ToolExecutionOutcome] = []
        for entry in prepared_entries:
            if isinstance(entry, _ToolExecutionOutcome):
                results.append(entry)
            else:
                results.append(
                    await _execute_prepared_tool_call(entry, state, config, emit, signal)
                )
    else:
        task_entries: list[tuple[int, asyncio.Task[_ToolExecutionOutcome]]] = []
        for entry in prepared_entries:
            if isinstance(entry, _PreparedToolCall):
                task_entries.append(
                    (
                        entry.index,
                        asyncio.create_task(
                            _execute_prepared_tool_call(entry, state, config, emit, signal)
                        ),
                    )
                )

        task_results = await asyncio.gather(*(task for _, task in task_entries))
        results_by_index = {outcome.index: outcome for outcome in task_results}
        results = []
        for entry in prepared_entries:
            if isinstance(entry, _ToolExecutionOutcome):
                results.append(entry)
            else:
                results.append(results_by_index[entry.index])

    results = sorted(results, key=lambda outcome: outcome.index)
    should_terminate = bool(results) and all(outcome.result.terminate for outcome in results)
    return results, should_terminate


async def _prepare_tool_call(
    index: int,
    tool_call: ToolCall,
    state: AgentState,
    config: AgentConfig,
    emit: EmitFn,
    signal: asyncio.Event | None = None,
    assistant_msg: AssistantMessage | None = None,
) -> _PreparedToolCall | _ToolExecutionOutcome:
    await _emit_tool_execution_start(state, emit, tool_call)
    agent_tool = state.tool_lookup().get(tool_call.name)
    if agent_tool is None:
        error = RuntimeError(f"Unknown tool: {tool_call.name}")
        result = AgentToolResult(content=str(error), details=error, is_error=True)
        await _emit_tool_execution_end(state, emit, tool_call, result, is_error=True)
        return _ToolExecutionOutcome(index, tool_call, result, is_error=True)

    try:
        arguments = agent_tool.prepare_and_validate(tool_call.arguments)
    except ToolValidationError as exc:
        result = AgentToolResult(content=str(exc), details=exc, is_error=True)
        await _emit_tool_execution_end(state, emit, tool_call, result, is_error=True)
        return _ToolExecutionOutcome(index, tool_call, result, is_error=True)
    except Exception as exc:
        result = AgentToolResult(content=str(exc), details=exc, is_error=True)
        await _emit_tool_execution_end(state, emit, tool_call, result, is_error=True)
        return _ToolExecutionOutcome(index, tool_call, result, is_error=True)

    _assistant_msg = assistant_msg or AssistantMessage(content=[])

    if config.before_tool_call is not None:
        try:
            ctx = BeforeToolCallContext(
                tool_call=tool_call,
                agent_tool=agent_tool,
                assistant_message=_assistant_msg,
                arguments=arguments,
                state=state,
            )
            before_result = config.before_tool_call(ctx)
            if inspect.isawaitable(before_result):
                before_result = await before_result
        except Exception as exc:
            result = AgentToolResult(content=str(exc), details=exc, is_error=True)
            await _emit_tool_execution_end(state, emit, tool_call, result, is_error=True)
            return _ToolExecutionOutcome(index, tool_call, result, is_error=True)

        if isinstance(before_result, BeforeToolCallResult):
            result = AgentToolResult(
                content=before_result.content,
                terminate=before_result.terminate,
                is_error=before_result.is_error,
            )
            await _emit_tool_execution_end(
                state,
                emit,
                tool_call,
                result,
                is_error=before_result.is_error,
            )
            return _ToolExecutionOutcome(index, tool_call, result, is_error=before_result.is_error)

    return _PreparedToolCall(index, tool_call, agent_tool, arguments, _assistant_msg)


async def _execute_prepared_tool_call(
    prepared: _PreparedToolCall,
    state: AgentState,
    config: AgentConfig,
    emit: EmitFn,
    signal: asyncio.Event | None = None,
) -> _ToolExecutionOutcome:
    update_tasks: list[asyncio.Task[Any]] = []

    def on_update(update: Any) -> None:
        update_tasks.append(
            asyncio.create_task(
                _emit(emit, AgentEvent.tool_execution_update(prepared.tool_call, update))
            )
        )

    result: AgentToolResult
    is_error = False
    try:
        result = await _execute_agent_tool(
            prepared.agent_tool,
            prepared.tool_call.id,
            prepared.arguments,
            signal,
            on_update,
        )
        is_error = _result_is_error(result)
    except ToolValidationError as exc:
        result = AgentToolResult(content=str(exc), details=exc, is_error=True)
        is_error = True
    except asyncio.CancelledError:
        if update_tasks:
            await asyncio.gather(*update_tasks, return_exceptions=True)
        raise
    except Exception as exc:
        result = AgentToolResult(content=str(exc), details=exc, is_error=True)
        is_error = True

    if update_tasks:
        await asyncio.gather(*update_tasks, return_exceptions=True)

    if config.after_tool_call is not None:
        try:
            ctx = AfterToolCallContext(
                tool_call=prepared.tool_call,
                agent_tool=prepared.agent_tool,
                result=result,
                is_error=is_error,
                assistant_message=prepared.assistant_message,
                arguments=prepared.arguments,
                state=state,
            )
            after_result = config.after_tool_call(ctx)
            if inspect.isawaitable(after_result):
                after_result = await after_result
            if isinstance(after_result, AgentToolResult):
                result = after_result
                is_error = _result_is_error(result)
        except Exception as exc:
            result = AgentToolResult(content=str(exc), details=exc, is_error=True)
            is_error = True

    await _emit_tool_execution_end(state, emit, prepared.tool_call, result, is_error)
    return _ToolExecutionOutcome(prepared.index, prepared.tool_call, result, is_error)


async def _execute_agent_tool(
    agent_tool: AgentTool,
    tool_call_id: str,
    arguments: dict[str, Any],
    signal: asyncio.Event | None,
    on_update: Callable[[Any], None],
) -> AgentToolResult:
    if type(agent_tool).execute_validated is not AgentTool.execute_validated:
        return await agent_tool.execute_validated(tool_call_id, arguments, signal, on_update)
    if type(agent_tool).execute is not AgentTool.execute:
        return await agent_tool.execute(tool_call_id, arguments, signal, on_update)
    return await agent_tool.execute_validated(tool_call_id, arguments, signal, on_update)


def _result_is_error(result: AgentToolResult) -> bool:
    if result.is_error is not None:
        return result.is_error
    return isinstance(result.details, Exception)


async def _emit_tool_execution_start(
    state: AgentState,
    emit: EmitFn,
    tool_call: ToolCall,
) -> None:
    if tool_call.id not in state.pending_tool_calls:
        state.pending_tool_calls.append(tool_call.id)
    await _emit(emit, AgentEvent.tool_execution_start(tool_call))


async def _emit_tool_execution_end(
    state: AgentState,
    emit: EmitFn,
    tool_call: ToolCall,
    result: AgentToolResult,
    is_error: bool,
) -> None:
    state.pending_tool_calls = [id for id in state.pending_tool_calls if id != tool_call.id]
    await _emit(emit, AgentEvent.tool_execution_end(tool_call, result, is_error))


def _execution_mode(tool_call: ToolCall, lookup: dict[str, AgentTool]) -> str:
    agent_tool = lookup.get(tool_call.name)
    return agent_tool.execution_mode if agent_tool is not None else "parallel"


def _build_run_options(base_options: RunOptions | None, signal: asyncio.Event | None) -> RunOptions:
    if base_options is None:
        return RunOptions(abort_signal=signal)
    return dataclasses.replace(base_options, abort_signal=signal)


async def _get_messages(getter: Callable[[], Any] | None) -> list[Message]:
    if getter is None:
        return []
    messages = getter()
    if inspect.isawaitable(messages):
        messages = await messages
    return list(messages or [])


async def _should_stop_after_turn(
    config: AgentConfig,
    context: ShouldStopAfterTurnContext,
) -> bool:
    callback = config.should_stop_after_turn
    if callback is None:
        return False

    try:
        signature = inspect.signature(callback)
        accepts_context = len(signature.parameters) > 0
    except (TypeError, ValueError):
        accepts_context = True

    stop = callback(context) if accepts_context else callback()
    if inspect.isawaitable(stop):
        stop = await stop
    return bool(stop)


async def _append_failure_message(
    error: Exception,
    state: AgentState,
    emit: EmitFn,
    signal: asyncio.Event | None = None,
) -> None:
    failure_message = _error_assistant_message(error, None, signal=signal)
    state.error_message = failure_message.error_message
    state.messages = [*state.messages, failure_message]
    await _emit(emit, AgentEvent.message_start(failure_message))
    await _emit(emit, AgentEvent.message_end(failure_message))


def _error_assistant_message(
    error: Exception,
    partial: Message | None,
    *,
    signal: asyncio.Event | None = None,
    aborted: bool = False,
) -> AssistantMessage:
    message = partial if isinstance(partial, AssistantMessage) else AssistantMessage(content=[])
    is_aborted = aborted or (signal is not None and signal.is_set())
    is_aborted = is_aborted or str(error) == StopReason.ABORTED.value
    message.stop_reason = StopReason.ABORTED.value if is_aborted else StopReason.ERROR.value
    message.error_message = str(error)
    return message


def _is_error_or_aborted(message: AssistantMessage) -> bool:
    return message.stop_reason in {StopReason.ERROR.value, StopReason.ABORTED.value}


async def _emit(emit: EmitFn, event: AgentEvent) -> None:
    result = emit(event)
    if inspect.isawaitable(result):
        await result
