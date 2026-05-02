from __future__ import annotations

from agentkit.llm import Context, ImageContent, Message, Role, TextContent, ThinkingContent, ToolCall, ToolResult
from agentkit.llm.providers.anthropic._convert import build_request as build_anthropic_request
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


def test_anthropic_thinking_signature_is_included_in_conversion() -> None:
    msg = Message(
        role=Role.ASSISTANT,
        content=[ThinkingContent(text="Let me think.", signature="sig123")],
    )

    converted = convert_anthropic_messages([msg])

    block = converted[0]["content"][0]
    assert block["type"] == "thinking"
    assert block["thinking"] == "Let me think."
    assert block["signature"] == "sig123"


def test_anthropic_thinking_without_signature_omits_field() -> None:
    msg = Message(
        role=Role.ASSISTANT,
        content=[ThinkingContent(text="Thinking.", signature=None)],
    )

    converted = convert_anthropic_messages([msg])

    block = converted[0]["content"][0]
    assert block["type"] == "thinking"
    assert block["thinking"] == "Thinking."
    assert "signature" not in block


def test_anthropic_redacted_thinking_converts_correctly() -> None:
    msg = Message(
        role=Role.ASSISTANT,
        content=[ThinkingContent(text="", signature="opaque-data", redacted=True)],
    )

    converted = convert_anthropic_messages([msg])

    block = converted[0]["content"][0]
    assert block["type"] == "redacted_thinking"
    assert block["data"] == "opaque-data"


def test_anthropic_cache_control_on_system_prompt() -> None:
    ctx = Context(system_prompt="You are helpful.")
    options = ModelOptions(model="claude-3-5-sonnet", cache_control="ephemeral")

    request = build_anthropic_request(ctx, options)

    assert isinstance(request["system"], list)
    assert request["system"][0]["type"] == "text"
    assert request["system"][0]["text"] == "You are helpful."
    assert request["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_no_cache_control_by_default() -> None:
    ctx = Context(system_prompt="You are helpful.")
    options = ModelOptions(model="claude-3-5-sonnet")

    request = build_anthropic_request(ctx, options)

    assert request["system"] == "You are helpful."


def test_anthropic_cache_control_on_last_tool() -> None:
    from agentkit.llm import Tool

    tools = [
        Tool(name="search", description="Search", parameters={"type": "object", "properties": {}}),
        Tool(name="calc", description="Calculate", parameters={"type": "object", "properties": {}}),
    ]
    ctx = Context(tools=tools)
    options = ModelOptions(model="claude-3-5-sonnet", tools=tools, cache_control="ephemeral")

    request = build_anthropic_request(ctx, options)

    assert "cache_control" not in request["tools"][0]
    assert request["tools"][-1]["cache_control"] == {"type": "ephemeral"}
