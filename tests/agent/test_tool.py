from __future__ import annotations

import asyncio

import pytest

from agentkit.agent import AgentTool, AgentToolResult, agent_tool
from agentkit.agent.tool import _coerce_to_tool_result
from agentkit.llm import TextContent, Tool, tool


def test_agent_tool_mirrors_wrapped_tool(echo_agent_tool: AgentTool) -> None:
    assert echo_agent_tool.name == echo_agent_tool.tool.name
    assert echo_agent_tool.description == echo_agent_tool.tool.description
    assert echo_agent_tool.parameters == echo_agent_tool.tool.parameters
    assert echo_agent_tool.display_label == echo_agent_tool.name
    assert echo_agent_tool.to_tool() is echo_agent_tool.tool


def test_agent_tool_display_label_uses_label(echo_agent_tool: AgentTool) -> None:
    echo_agent_tool.label = "Echo"

    assert echo_agent_tool.display_label == "Echo"


async def test_execute_sync_tool(echo_agent_tool: AgentTool) -> None:
    result = await echo_agent_tool.execute("tc", {"value": "hi"})

    assert result == AgentToolResult(content="echo: hi")


async def test_execute_async_tool() -> None:
    @tool()
    async def uppercase(text: str) -> str:
        return text.upper()

    result = await AgentTool(uppercase).execute("tc", {"text": "hello"})

    assert result.content == "HELLO"


async def test_execute_tool_returning_agent_tool_result() -> None:
    raw = AgentToolResult("done", terminate=True)

    @tool()
    def finish() -> AgentToolResult:
        return raw

    result = await AgentTool(finish).execute("tc", {})

    assert result is raw


async def test_execute_with_preset_signal_raises_cancelled(echo_agent_tool: AgentTool) -> None:
    signal = asyncio.Event()
    signal.set()

    with pytest.raises(asyncio.CancelledError):
        await echo_agent_tool.execute("tc", {"value": "hi"}, signal)


def test_execute_validation_error_returns_tool_result(echo_agent_tool: AgentTool) -> None:
    async def run() -> AgentToolResult:
        return await echo_agent_tool.execute("tc", {})

    result = asyncio.run(run())

    assert "Invalid arguments for tool echo" in result.content
    assert result.details is not None


def test_coerce_to_tool_result_variants() -> None:
    existing = AgentToolResult("ok")
    text_list = [TextContent(text="ok")]
    obj = {"answer": 42}

    assert _coerce_to_tool_result(existing) is existing
    assert _coerce_to_tool_result("ok") == AgentToolResult("ok")
    assert _coerce_to_tool_result(text_list) == AgentToolResult(text_list)
    assert _coerce_to_tool_result(obj) == AgentToolResult(str(obj), details=obj)


def test_agent_tool_decorator_creates_agent_tool() -> None:
    @agent_tool(label="Search", execution_mode="sequential")
    def search(query: str) -> str:
        """Search docs."""
        return query

    assert isinstance(search, AgentTool)
    assert search.name == "search"
    assert search.description == "Search docs."
    assert search.display_label == "Search"
    assert search.execution_mode == "sequential"


def test_agent_tool_wraps_external_tool() -> None:
    external = Tool(name="external", description="", parameters={"type": "object"})

    assert AgentTool(external).to_tool() is external


async def test_prepare_arguments_applied_before_execution() -> None:
    received: list[dict] = []

    @tool()
    def capture(value: str) -> str:
        received.append({"value": value})
        return value

    def normalise(args: dict) -> dict:
        return {k: v.upper() for k, v in args.items()}

    agent = AgentTool(capture, prepare_arguments=normalise)
    result = await agent.execute("tc", {"value": "hello"})

    assert result.content == "HELLO"
    assert received[0]["value"] == "HELLO"
