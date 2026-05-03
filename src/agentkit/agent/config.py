from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from agentkit.llm.model import RunOptions
from agentkit.llm.types import Message

from .types import (
    AfterToolCallContext,
    AgentToolResult,
    BeforeToolCallContext,
    BeforeToolCallResult,
    ContextTransformer,
    ExecutionMode,
    ShouldStopAfterTurnContext,
)

__all__ = ["AgentConfig"]


MessageGetter = Callable[[], Awaitable[list[Message]] | list[Message]]


@dataclass(frozen=True, slots=True)
class AgentConfig:
    max_turns: int | None = None
    transform_context: ContextTransformer | None = None
    before_tool_call: (
        Callable[
            [BeforeToolCallContext],
            Awaitable[BeforeToolCallResult | None] | BeforeToolCallResult | None,
        ]
        | None
    ) = None
    after_tool_call: (
        Callable[
            [AfterToolCallContext],
            Awaitable[AgentToolResult | None] | AgentToolResult | None,
        ]
        | None
    ) = None
    should_stop_after_turn: (
        Callable[[ShouldStopAfterTurnContext], Awaitable[bool] | bool]
        | Callable[[], Awaitable[bool] | bool]
        | None
    ) = None
    steering_mode: Literal["all", "one-at-a-time"] = "one-at-a-time"
    follow_up_mode: Literal["all", "one-at-a-time"] = "one-at-a-time"
    tool_execution: ExecutionMode = "parallel"
    run_options: RunOptions | None = None
    _get_steer_messages: MessageGetter | None = None
    _get_follow_up_messages: MessageGetter | None = None
