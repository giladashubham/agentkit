from __future__ import annotations

from agentkit.llm import Context, ImageContent, Message, Role, TextContent, ToolCall, ToolResult
from agentkit.llm.providers.anthropic._convert import convert_messages as convert_anthropic_messages
from agentkit.llm.providers.base import ModelOptions
from agentkit.llm.providers.google._convert import build_request as build_google_request
from agentkit.llm.providers.google._convert import convert_messages as convert_google_messages
from agentkit.llm.providers.openai._convert import convert_content as convert_openai_content
from agentkit.llm.providers.openai_responses._convert import (
    build_request as build_responses_request,
)
from agentkit.llm.providers.openai_ws._convert import convert_context_to_input


class _Types:
    class GenerateContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs


def test_timeout_ms_is_converted_to_seconds_for_openai_responses() -> None:
    request = build_responses_request(Context(), ModelOptions(model="gpt", timeout_ms=30_000))

    assert request["timeout"] == 30


def test_google_extra_config_does_not_mutate_options() -> None:
    options = ModelOptions(model="gemini", extra={"config": {"candidate_count": 1}, "labels": {}})

    request = build_google_request(Context(), options, _Types)

    assert request["config"].kwargs["candidate_count"] == 1
    assert request["labels"] == {}
    assert options.extra == {"config": {"candidate_count": 1}, "labels": {}}


def test_anthropic_tool_results_are_sent_as_user_messages() -> None:
    msg = Message(
        role=Role.TOOL_RESULT,
        content=[ToolResult(tool_call_id="call_1", tool_name="search", content="result")],
    )

    converted = convert_anthropic_messages([msg])

    assert converted[0]["role"] == "user"
    assert converted[0]["content"][0]["type"] == "tool_result"


def test_image_content_is_converted_for_multimodal_providers() -> None:
    image = ImageContent(data="abc", mimeType="image/png")

    assert convert_openai_content([TextContent(text="see"), image]) == [
        {"type": "text", "text": "see"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
    ]

    google = convert_google_messages([Message(role=Role.USER, content=[image])])
    assert google[0]["parts"] == [{"inline_data": {"mime_type": "image/png", "data": "abc"}}]

    context = Context(messages=[Message(role=Role.USER, content=[image])])
    responses = convert_context_to_input(context)
    assert responses[0]["content"] == [
        {"type": "input_image", "image_url": "data:image/png;base64,abc"}
    ]


def test_openai_responses_input_still_uses_simple_string_for_text_only() -> None:
    ctx = Context().add_user("hello")

    assert convert_context_to_input(ctx) == [
        {"type": "message", "role": "user", "content": "hello"}
    ]


def test_anthropic_preserves_assistant_tool_calls() -> None:
    msg = Message(
        role=Role.ASSISTANT,
        content=[ToolCall(id="call_1", name="search", arguments={"q": "x"})],
    )

    converted = convert_anthropic_messages([msg])

    assert converted[0]["role"] == "assistant"
    assert converted[0]["content"][0]["type"] == "tool_use"
