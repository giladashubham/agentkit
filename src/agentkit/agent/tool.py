from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from agentkit.exceptions import ToolValidationError
from agentkit.llm.tools import Tool, execute_tool, validate_tool_arguments
from agentkit.llm.tools import tool as llm_tool
from agentkit.llm.types import ImageContent, TextContent

from .types import AgentToolResult, ExecutionMode

__all__ = ["AgentTool", "agent_tool"]


@dataclass(slots=True)
class AgentTool:
    tool: Tool
    label: str | None = None
    execution_mode: ExecutionMode = "parallel"
    prepare_arguments: Callable[[Any], Any] | None = None

    @property
    def name(self) -> str:
        return self.tool.name

    @property
    def description(self) -> str:
        return self.tool.description

    @property
    def parameters(self) -> dict[str, Any]:
        return self.tool.parameters

    @property
    def display_label(self) -> str:
        return self.label or self.name

    def to_tool(self) -> Tool:
        return self.tool

    def prepare(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.prepare_arguments is None:
            return arguments
        return self.prepare_arguments(arguments)

    def validate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return validate_tool_arguments(self.tool, arguments)

    def prepare_and_validate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.validate(self.prepare(arguments))

    async def execute(
        self,
        tool_call_id: str,
        arguments: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update: Callable[[Any], None] | None = None,
    ) -> AgentToolResult:
        try:
            actual_args = self.prepare_and_validate(arguments)
        except ToolValidationError as exc:
            return AgentToolResult(content=str(exc), details=exc)
        return await self.execute_validated(tool_call_id, actual_args, signal, on_update)

    async def execute_validated(
        self,
        tool_call_id: str,
        arguments: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update: Callable[[Any], None] | None = None,
    ) -> AgentToolResult:
        del tool_call_id, on_update
        if signal is not None and signal.is_set():
            raise asyncio.CancelledError

        try:
            raw = await execute_tool(self.tool, arguments, validate=False)
        except ToolValidationError as exc:
            return AgentToolResult(content=str(exc), details=exc)
        if signal is not None and signal.is_set():
            raise asyncio.CancelledError
        return _coerce_to_tool_result(raw)


def agent_tool(
    name: str | Callable[..., Any] | None = None,
    description: str | None = None,
    label: str | None = None,
    execution_mode: ExecutionMode = "parallel",
    prepare_arguments: Callable[[Any], Any] | None = None,
) -> Callable[[Callable[..., Any]], AgentTool] | AgentTool:
    """Decorator to create an AgentTool from a Python function."""
    if callable(name):
        func = name
        return AgentTool(
            tool=llm_tool()(func),
            label=label,
            execution_mode=execution_mode,
            prepare_arguments=prepare_arguments,
        )

    def decorator(func: Callable[..., Any]) -> AgentTool:
        wrapped = llm_tool(name=cast(str | None, name), description=description)(func)
        return AgentTool(
            tool=wrapped,
            label=label,
            execution_mode=execution_mode,
            prepare_arguments=prepare_arguments,
        )

    return decorator


def _coerce_to_tool_result(raw: Any) -> AgentToolResult:
    if isinstance(raw, AgentToolResult):
        return raw
    if isinstance(raw, str):
        return AgentToolResult(content=raw)
    if isinstance(raw, list) and all(isinstance(item, TextContent | ImageContent) for item in raw):
        return AgentToolResult(content=raw)
    return AgentToolResult(content=str(raw), details=raw)
