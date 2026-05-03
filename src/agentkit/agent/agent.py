from __future__ import annotations

import asyncio
import dataclasses
import inspect
from collections.abc import Callable
from typing import Literal, cast

from agentkit.llm.model import Model
from agentkit.llm.types import Message, Role

from .config import AgentConfig
from .loop import run_agent_loop
from .state import AgentState
from .tool import AgentTool
from .types import AgentEvent, AgentListener, StreamFn

__all__ = ["Agent"]


class Agent:
    def __init__(
        self,
        model: Model,
        system_prompt: str | None = None,
        tools: list[AgentTool] | None = None,
        config: AgentConfig | None = None,
        stream_fn: StreamFn | None = None,
    ):
        self._model = model
        self._system_prompt = system_prompt
        self._tools = list(tools or [])
        self._config = config or AgentConfig()
        self._stream_fn = stream_fn

        self._state = AgentState(
            model=self._model,
            system_prompt=self._system_prompt,
            _tools=self._tools,
            run_options=self._config.run_options,
        )
        self._listeners: list[AgentListener] = []
        self._run_task: asyncio.Task[None] | None = None
        self._abort_signal: asyncio.Event | None = None
        self._idle_event = asyncio.Event()
        self._idle_event.set()
        self._steer_queue: asyncio.Queue[list[Message]] = asyncio.Queue()
        self._follow_up_queue: asyncio.Queue[list[Message]] = asyncio.Queue()

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._run_task is not None and not self._run_task.done()

    def subscribe(self, listener: AgentListener) -> Callable[[], None]:
        self._listeners.append(listener)

        def unsubscribe() -> None:
            try:
                self._listeners.remove(listener)
            except ValueError:
                pass

        return unsubscribe

    async def run(
        self,
        prompt: str | list[Message],
        *,
        system_prompt: str | None = None,
        tools: list[AgentTool] | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        """Add a user prompt or messages, then run until the agent becomes idle."""
        await self._run_messages(
            _normalize_messages(prompt),
            system_prompt=system_prompt,
            tools=tools,
            config=config,
        )

    async def steer(self, prompt: str | list[Message]) -> None:
        await self._steer_queue.put(_normalize_messages(prompt))

    async def follow_up(self, prompt: str | list[Message]) -> None:
        await self._follow_up_queue.put(_normalize_messages(prompt))

    async def cancel(self) -> None:
        if self._abort_signal is not None:
            self._abort_signal.set()

    async def wait_for_idle(self) -> None:
        await self._idle_event.wait()

    async def reset(self) -> None:
        """Cancel any running loop and clear all state and queued messages."""
        await self.cancel()
        await self.wait_for_idle()
        self.clear_all_queues()
        self._state.messages = []
        self._state.turn_index = 0
        self._state.is_streaming = False
        self._state.streaming_message = None
        self._state.pending_tool_calls = []
        self._state.error_message = None

    async def continue_(self, prompt: str | list[Message] | None = None) -> None:
        """Continue from existing context, or add a prompt and run when one is supplied."""
        if prompt is not None:
            await self.run(prompt)
            return

        if self.is_running:
            raise RuntimeError("Agent is already running")

        state = self._state
        if not state.messages:
            raise RuntimeError("No messages to continue from")

        last_message = state.messages[-1]
        if last_message.role == Role.ASSISTANT:
            queued_steering = _drain_queue(self._steer_queue, self._config.steering_mode)
            if queued_steering:
                await self._run_messages(queued_steering, skip_initial_steer=True)
                return

            queued_follow_ups = _drain_queue(self._follow_up_queue, self._config.follow_up_mode)
            if queued_follow_ups:
                await self._run_messages(queued_follow_ups)
                return

            raise RuntimeError("Cannot continue from message role: assistant")

        await self._run_messages([])

    def clear_steering_queue(self) -> None:
        _drain_queue_discard(self._steer_queue)

    def clear_follow_up_queue(self) -> None:
        _drain_queue_discard(self._follow_up_queue)

    def clear_all_queues(self) -> None:
        self.clear_steering_queue()
        self.clear_follow_up_queue()

    def has_queued_messages(self) -> bool:
        return not self._steer_queue.empty() or not self._follow_up_queue.empty()

    async def _run_messages(
        self,
        initial_messages: list[Message],
        *,
        system_prompt: str | None = None,
        tools: list[AgentTool] | None = None,
        config: AgentConfig | None = None,
        skip_initial_steer: bool = False,
    ) -> None:
        if self.is_running:
            raise RuntimeError("Agent is already running")

        run_config = self._make_config(
            config or self._config,
            skip_initial_steer=skip_initial_steer,
        )
        state = self._ensure_state()
        if system_prompt is not None:
            state.system_prompt = system_prompt
        if tools is not None:
            state.tools = tools
        state.run_options = run_config.run_options
        state.error_message = None

        self._abort_signal = asyncio.Event()
        state.is_streaming = True
        state.streaming_message = None
        state.pending_tool_calls = []
        self._idle_event.clear()
        current_task = asyncio.current_task()
        if current_task is None:
            raise RuntimeError("Agent.run() must be called from an asyncio task")
        self._run_task = cast(asyncio.Task[None], current_task)

        await self._run_loop(initial_messages, state, run_config)

    async def _run_loop(
        self,
        initial_messages: list[Message],
        state: AgentState,
        config: AgentConfig,
    ) -> None:
        async def emit(event: AgentEvent) -> None:
            for listener in list(self._listeners):
                try:
                    result = listener(event)
                    if inspect.isawaitable(result):
                        await result
                except Exception:
                    pass

        try:
            await run_agent_loop(
                initial_messages,
                state,
                config,
                emit,
                self._abort_signal,
                self._stream_fn,
            )
        except asyncio.CancelledError:
            state.error_message = "cancelled"
        except Exception as exc:
            state.error_message = str(exc)
        finally:
            self._idle_event.set()
            self._run_task = None

    def _ensure_state(self) -> AgentState:
        return self._state

    def _make_config(
        self,
        base_config: AgentConfig,
        *,
        skip_initial_steer: bool = False,
    ) -> AgentConfig:
        steering_mode = base_config.steering_mode
        follow_up_mode = base_config.follow_up_mode
        skip_steer = skip_initial_steer

        def get_steer_messages() -> list[Message]:
            nonlocal skip_steer
            if skip_steer:
                skip_steer = False
                return []
            return _drain_queue(self._steer_queue, steering_mode)

        return dataclasses.replace(
            base_config,
            _get_steer_messages=get_steer_messages,
            _get_follow_up_messages=lambda: _drain_queue(self._follow_up_queue, follow_up_mode),
        )


def _normalize_messages(prompt: str | list[Message]) -> list[Message]:
    if isinstance(prompt, str):
        return [Message.user(prompt)]
    return list(prompt)


def _drain_queue(
    queue: asyncio.Queue[list[Message]],
    mode: Literal["all", "one-at-a-time"] = "one-at-a-time",
) -> list[Message]:
    messages: list[Message] = []
    if mode == "one-at-a-time":
        try:
            messages.extend(queue.get_nowait())
        except asyncio.QueueEmpty:
            pass
    else:
        while True:
            try:
                messages.extend(queue.get_nowait())
            except asyncio.QueueEmpty:
                break
    return messages


def _drain_queue_discard(queue: asyncio.Queue[list[Message]]) -> None:
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            break
