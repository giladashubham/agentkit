from __future__ import annotations

from dataclasses import dataclass, field

from agentkit.llm.model import Model, RunOptions
from agentkit.llm.tools import Tool
from agentkit.llm.types import AssistantMessage, Message

from .tool import AgentTool

__all__ = ["AgentState"]


@dataclass(slots=True, init=False)
class AgentState:
    model: Model
    system_prompt: str | None = None
    _tools: list[AgentTool] = field(default_factory=list, repr=False)
    _messages: list[Message] = field(default_factory=list, repr=False)
    turn_index: int = 0
    is_streaming: bool = False
    streaming_message: AssistantMessage | None = None
    pending_tool_calls: list[str] = field(default_factory=list)
    error_message: str | None = None
    run_options: RunOptions | None = None

    def __init__(
        self,
        model: Model,
        system_prompt: str | None = None,
        tools: list[AgentTool] | None = None,
        messages: list[Message] | None = None,
        *,
        _tools: list[AgentTool] | None = None,
        _messages: list[Message] | None = None,
        turn_index: int = 0,
        is_streaming: bool = False,
        streaming_message: AssistantMessage | None = None,
        pending_tool_calls: list[str] | None = None,
        error_message: str | None = None,
        run_options: RunOptions | None = None,
    ) -> None:
        if tools is not None and _tools is not None:
            raise TypeError("Pass either tools or _tools, not both")
        if messages is not None and _messages is not None:
            raise TypeError("Pass either messages or _messages, not both")

        self.model = model
        self.system_prompt = system_prompt
        self._tools = list(tools if tools is not None else _tools or [])
        self._messages = list(messages if messages is not None else _messages or [])
        self.turn_index = turn_index
        self.is_streaming = is_streaming
        self.streaming_message = streaming_message
        self.pending_tool_calls = list(pending_tool_calls or [])
        self.error_message = error_message
        self.run_options = run_options

    @property
    def tools(self) -> list[AgentTool]:
        return self._tools

    @tools.setter
    def tools(self, value: list[AgentTool]) -> None:
        self._tools = list(value)

    @property
    def messages(self) -> list[Message]:
        return self._messages

    @messages.setter
    def messages(self, value: list[Message]) -> None:
        self._messages = list(value)

    def tool_lookup(self) -> dict[str, AgentTool]:
        return {tool.name: tool for tool in self._tools}

    def llm_tools(self) -> list[Tool]:
        return [tool.to_tool() for tool in self._tools]
