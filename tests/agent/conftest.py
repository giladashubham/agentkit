from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable

import pytest

from agentkit.agent import AgentTool
from agentkit.llm import (
    AssistantMessage,
    Context,
    Model,
    Response,
    RunOptions,
    StopReason,
    StreamEvent,
    StreamResponse,
    TextContent,
    ToolCall,
    Usage,
    tool,
)


@pytest.fixture
def mock_model() -> Model:
    return Model(id="test", provider="test", api="test-api")


@pytest.fixture
def echo_agent_tool() -> AgentTool:
    @tool()
    def echo(value: str) -> str:
        return f"echo: {value}"

    return AgentTool(echo)


def make_text_stream(
    text: str,
    tool_calls: list[ToolCall] | None = None,
    *,
    delay: float = 0,
) -> Callable[[Model, Context, RunOptions | None], StreamResponse]:
    def stream_fn(model: Model, _context: Context, _options: RunOptions | None) -> StreamResponse:
        async def events() -> AsyncIterator[StreamEvent]:
            if delay:
                await asyncio.sleep(delay)
            yield StreamEvent.start(model.id)
            yield StreamEvent.text_start(partial=AssistantMessage(content=[]))
            content = []
            if text:
                content.append(TextContent(text=text))
                partial = AssistantMessage(content=[TextContent(text=text)])
                yield StreamEvent.text_delta(text, partial=partial)
            else:
                partial = AssistantMessage(content=[])
            if tool_calls:
                content.extend(tool_calls)
                partial = AssistantMessage(content=content)
                for call in tool_calls:
                    yield StreamEvent.toolcall_end(call, partial=partial)
            yield StreamEvent.done(
                Response(
                    message=partial, stop_reason=StopReason.STOP, usage=Usage(), model=model.id
                )
            )

        return StreamResponse(events())

    return stream_fn
