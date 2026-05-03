from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

from agentkit.llm.context import Context
from agentkit.llm.model import Model, RunOptions
from agentkit.llm.streaming import StreamEvent, StreamResponse
from agentkit.llm.types import AssistantMessage, ImageContent, Message, TextContent, ToolCall

if TYPE_CHECKING:
    from .state import AgentState
    from .tool import AgentTool

__all__ = [
    "AgentEventType",
    "AgentEvent",
    "AgentToolResult",
    "BeforeToolCallContext",
    "BeforeToolCallResult",
    "AfterToolCallContext",
    "ShouldStopAfterTurnContext",
    "ExecutionMode",
    "AgentListener",
    "ContextTransformer",
    "StreamFn",
]

ToolResultContent = str | list[TextContent | ImageContent]


class AgentEventType(StrEnum):
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    TURN_START = "turn_start"
    TURN_END = "turn_end"
    MESSAGE_START = "message_start"
    MESSAGE_UPDATE = "message_update"
    MESSAGE_END = "message_end"
    TOOL_EXECUTION_START = "tool_execution_start"
    TOOL_EXECUTION_UPDATE = "tool_execution_update"
    TOOL_EXECUTION_END = "tool_execution_end"


@dataclass(slots=True)
class AgentEvent:
    type: AgentEventType
    data: Any = None

    @classmethod
    def agent_start(cls) -> AgentEvent:
        return cls(AgentEventType.AGENT_START)

    @classmethod
    def agent_end(cls, messages: list[Message] | None = None) -> AgentEvent:
        return cls(AgentEventType.AGENT_END, messages or [])

    @classmethod
    def turn_start(cls, turn_index: int) -> AgentEvent:
        return cls(AgentEventType.TURN_START, turn_index)

    @classmethod
    def turn_end(
        cls,
        turn_index: int,
        assistant_message: AssistantMessage,
        tool_result_messages: list[Message],
    ) -> AgentEvent:
        return cls(AgentEventType.TURN_END, (turn_index, assistant_message, tool_result_messages))

    @classmethod
    def message_start(cls, message: Message) -> AgentEvent:
        return cls(AgentEventType.MESSAGE_START, message)

    @classmethod
    def message_update(
        cls, message: AssistantMessage, raw_event: StreamEvent | None = None
    ) -> AgentEvent:
        return cls(AgentEventType.MESSAGE_UPDATE, (message, raw_event))

    @classmethod
    def message_end(cls, message: Message) -> AgentEvent:
        return cls(AgentEventType.MESSAGE_END, message)

    @classmethod
    def tool_execution_start(cls, tool_call: ToolCall) -> AgentEvent:
        return cls(AgentEventType.TOOL_EXECUTION_START, tool_call)

    @classmethod
    def tool_execution_update(cls, tool_call: ToolCall, update: Any) -> AgentEvent:
        return cls(AgentEventType.TOOL_EXECUTION_UPDATE, (tool_call, update))

    @classmethod
    def tool_execution_end(
        cls, tool_call: ToolCall, result: AgentToolResult, is_error: bool = False
    ) -> AgentEvent:
        return cls(AgentEventType.TOOL_EXECUTION_END, (tool_call, result, is_error))


@dataclass(slots=True)
class AgentToolResult:
    content: ToolResultContent
    details: Any = None
    terminate: bool = False
    is_error: bool | None = None


@dataclass(slots=True)
class BeforeToolCallResult:
    content: ToolResultContent
    is_error: bool = False
    terminate: bool = False


@dataclass(slots=True)
class BeforeToolCallContext:
    tool_call: ToolCall
    agent_tool: AgentTool
    assistant_message: AssistantMessage
    arguments: dict[str, Any]
    state: AgentState

    @property
    def args(self) -> dict[str, Any]:
        """Alias matching Pi's TypeScript beforeToolCall context name."""
        return self.arguments


@dataclass(slots=True)
class AfterToolCallContext:
    tool_call: ToolCall
    agent_tool: AgentTool
    result: AgentToolResult
    is_error: bool
    assistant_message: AssistantMessage
    arguments: dict[str, Any]
    state: AgentState

    @property
    def args(self) -> dict[str, Any]:
        """Alias matching Pi's TypeScript afterToolCall context name."""
        return self.arguments


@dataclass(slots=True)
class ShouldStopAfterTurnContext:
    assistant_message: AssistantMessage
    tool_result_messages: list[Message]
    state: AgentState
    new_messages: list[Message]

    @property
    def message(self) -> AssistantMessage:
        """Alias matching Pi's TypeScript shouldStopAfterTurn context name."""
        return self.assistant_message

    @property
    def tool_results(self) -> list[Message]:
        """Alias matching Pi's TypeScript shouldStopAfterTurn context name."""
        return self.tool_result_messages


ExecutionMode = Literal["parallel", "sequential"]

AgentListener = Callable[[AgentEvent], Awaitable[None] | None]
ContextTransformer = Callable[[list[Message]], Awaitable[list[Message]] | list[Message]]
StreamFn = Callable[[Model, Context, RunOptions | None], StreamResponse]
