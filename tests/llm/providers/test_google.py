from __future__ import annotations

from types import SimpleNamespace

from agentkit.llm import Context, Message, Role, ToolCall, ToolResult, tool
from agentkit.llm.providers.google._convert import (
    convert_messages,
    convert_tool,
    convert_tool_choice,
)
from agentkit.llm.providers.google._parse import parse_response


def test_google_convert_user_and_assistant_tool_call() -> None:
    ctx = Context()
    ctx.add_user("hello")
    ctx.add_message(Message(
        role=Role.ASSISTANT,
        content=[ToolCall(id="call_1", name="search", arguments={"q": "x"})],
    ))

    converted = convert_messages(ctx.messages)

    assert converted[0] == {"role": "user", "parts": [{"text": "hello"}]}
    assert converted[1]["role"] == "model"
    assert converted[1]["parts"][0]["function_call"]["name"] == "search"


def test_google_convert_tool_result() -> None:
    msg = Message(
        role=Role.TOOL_RESULT,
        content=[ToolResult(tool_call_id="call_1", tool_name="search", content="result")],
    )

    converted = convert_messages([msg])

    response = converted[0]["parts"][0]["function_response"]
    assert response["id"] == "call_1"
    assert response["name"] == "search"
    assert response["response"]["result"] == "result"


def test_google_tool_conversion() -> None:
    @tool()
    def search(query: str) -> str:
        """Search."""
        return query

    converted = convert_tool(search)

    declaration = converted["function_declarations"][0]
    assert declaration["name"] == "search"
    assert "query" in declaration["parameters"]["properties"]
    assert convert_tool_choice("auto") == {"function_calling_config": {"mode": "AUTO"}}
    assert convert_tool_choice("search") == {
        "function_calling_config": {
            "mode": "ANY",
            "allowed_function_names": ["search"],
        }
    }


def test_google_parse_response() -> None:
    response = SimpleNamespace(
        text="hello",
        function_calls=[SimpleNamespace(id="call_1", name="search", args={"q": "x"})],
        usage_metadata=SimpleNamespace(prompt_token_count=3, candidates_token_count=4),
        candidates=[SimpleNamespace(finish_reason="STOP")],
        model_version="gemini-2.5-flash",
    )

    parsed = parse_response(response, "gemini-2.5-flash")

    assert parsed.text() == "hello"
    assert parsed.tool_calls()[0].arguments == {"q": "x"}
    assert parsed.usage.input == 3
    assert parsed.usage.output == 4


def test_google_apis_are_registered() -> None:
    from agentkit.llm import list_provider_apis

    assert "google-generative-ai" in list_provider_apis()
    assert "google-vertex" in list_provider_apis()
