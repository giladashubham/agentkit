from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from agentkit.llm import (
    AssistantMessage,
    EventType,
    Model,
    ModelCost,
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


async def test_stream_result_returns_error_response_for_error_event() -> None:
    async def events() -> AsyncIterator[StreamEvent]:
        partial = AssistantMessage(content=[TextContent(text="partial")])
        yield StreamEvent.start("test")
        yield StreamEvent.text_delta("partial", partial=partial)
        yield StreamEvent.error(RuntimeError("boom"), partial=partial)

    stream = StreamResponse(events())

    response = await stream.result()

    assert response.stop_reason == StopReason.ERROR
    assert response.model == "test"
    assert response.text() == "partial"
    assert isinstance(response.raw, RuntimeError)


async def test_stream_iteration_converts_raised_errors_to_error_events() -> None:
    async def events() -> AsyncIterator[StreamEvent]:
        yield StreamEvent.start("test")
        raise RuntimeError("boom")

    stream = StreamResponse(events())

    events_seen = [event async for event in stream]
    response = await stream.result()

    assert events_seen[-1].type == EventType.ERROR
    assert response.stop_reason == StopReason.ERROR
    assert response.model == "test"


async def test_stream_result_maps_abort_errors_to_aborted_response() -> None:
    async def events() -> AsyncIterator[StreamEvent]:
        yield StreamEvent.start("test")
        raise RuntimeError(StopReason.ABORTED.value)

    stream = StreamResponse(events())

    response = await stream.result()

    assert response.stop_reason == StopReason.ABORTED


async def test_stream_result_calculates_cost_when_model_ref_is_available() -> None:
    model = Model(
        provider="custom",
        api="custom-api",
        id="test",
        cost=ModelCost(input=1.00, output=2.00),
    )

    async def events() -> AsyncIterator[StreamEvent]:
        yield StreamEvent.done(
            Response(
                message=AssistantMessage(content=[TextContent(text="done")]),
                stop_reason=StopReason.STOP,
                usage=Usage(input=100, output=50),
                model="test",
            )
        )

    stream = StreamResponse(events(), model_ref=model)

    response = await stream.result()

    assert response.usage.cost.input == pytest.approx(0.0001)
    assert response.usage.cost.output == pytest.approx(0.0001)
    assert response.usage.cost.total == pytest.approx(0.0002)
    assert response.message.provider == "custom"
    assert response.message.api == "custom-api"
    assert response.message.model == "test"
    assert response.message.usage == response.usage.model_dump(mode="json")
    assert response.message.stop_reason == "stop"
