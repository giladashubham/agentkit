from __future__ import annotations

from collections.abc import AsyncIterator

from agentkit.llm import (
    AssistantMessage,
    Response,
    Role,
    StopReason,
    StreamEvent,
    StreamResponse,
    TextContent,
    Usage,
)


async def _events() -> AsyncIterator[StreamEvent]:
    partial = AssistantMessage(content=[TextContent(text="hel")])
    yield StreamEvent.text_start(content_index=0, partial=AssistantMessage(content=[]))
    yield StreamEvent.text_delta("hel", content_index=0, partial=partial)
    partial = AssistantMessage(content=[TextContent(text="hello")])
    yield StreamEvent.text_delta("lo", content_index=0, partial=partial)
    yield StreamEvent.text_end("hello", content_index=0, partial=partial)
    yield StreamEvent.done(
        Response(
            message=partial,
            stop_reason=StopReason.STOP,
            usage=Usage(),
            model="test",
        )
    )


async def test_stream_result_returns_final_response() -> None:
    stream = StreamResponse(_events())

    response = await stream.result()

    assert response.text() == "hello"
    assert response.message.role == Role.ASSISTANT


async def test_stream_text_consumes_deltas() -> None:
    stream = StreamResponse(_events())

    assert await stream.text() == "hello"


async def test_stream_events_include_partial_and_content_index() -> None:
    stream = StreamResponse(_events())

    events = [event async for event in stream]

    assert events[1].content_index == 0
    assert events[1].partial is not None
    assert events[1].partial.text() == "hel"
