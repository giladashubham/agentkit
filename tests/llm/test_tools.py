from __future__ import annotations

import pytest
from pydantic import BaseModel

from agentkit.exceptions import ToolValidationError
from agentkit.llm import Tool, execute_tool, tool, tool_from_pydantic, validate_tool_arguments


def test_tool_decorator_generates_json_schema() -> None:
    @tool(description="Get the weather")
    def get_weather(city: str, unit: str = "celsius") -> str:
        """Get weather for a city."""
        return f"Weather in {city}"

    assert get_weather.name == "get_weather"
    assert get_weather.description == "Get the weather"
    assert get_weather.parameters == {
        "properties": {
            "city": {"title": "City", "type": "string"},
            "unit": {"default": "celsius", "title": "Unit", "type": "string"},
        },
        "required": ["city"],
        "type": "object",
    }


def test_tool_decorator_uses_docstring_description() -> None:
    @tool()
    def search(query: str) -> str:
        """Search the web."""
        return "results"

    assert search.description == "Search the web."


def test_tool_from_pydantic_model() -> None:
    class SearchParams(BaseModel):
        """Search parameters."""

        query: str
        max_results: int = 5

    search = tool_from_pydantic(SearchParams, name="search")

    assert search.name == "search"
    assert search.description == "Search parameters."
    assert search.parameters["required"] == ["query"]
    assert search.parameters["properties"]["max_results"]["default"] == 5


def test_tool_to_anthropic_shape(search_tool: Tool) -> None:
    assert search_tool.to_anthropic() == {
        "name": "search",
        "description": "Search the web for information.",
        "input_schema": search_tool.parameters,
    }


def test_tool_to_openai_shape(search_tool: Tool) -> None:
    assert search_tool.to_openai() == {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web for information.",
            "parameters": search_tool.parameters,
        },
    }


async def test_execute_sync_tool(search_tool: Tool) -> None:
    result = await execute_tool(search_tool, {"query": "python", "max_results": 3})

    assert result == "results for python"


async def test_execute_async_tool() -> None:
    @tool()
    async def uppercase(text: str) -> str:
        return text.upper()

    assert await execute_tool(uppercase, {"text": "hello"}) == "HELLO"


async def test_execute_tool_without_callable_raises() -> None:
    external_tool = Tool(name="external", description="", parameters={"type": "object"})

    with pytest.raises(ValueError, match="Tool external has no callable function"):
        await execute_tool(external_tool, {})


def test_validate_tool_arguments_coerces_decorator_arguments() -> None:
    @tool()
    def repeat(count: int, loud: bool = False) -> str:
        return "ok"

    assert validate_tool_arguments(repeat, {"count": "3", "loud": "true"}) == {
        "count": 3,
        "loud": True,
    }


async def test_execute_tool_validates_and_coerces_by_default() -> None:
    @tool()
    def repeat(count: int) -> int:
        return count + 1

    assert await execute_tool(repeat, {"count": "3"}) == 4


def test_validate_tool_arguments_raises_clear_error() -> None:
    @tool()
    def repeat(count: int) -> str:
        return "ok"

    with pytest.raises(ToolValidationError) as exc_info:
        validate_tool_arguments(repeat, {"count": "not-int"})

    assert exc_info.value.tool_name == "repeat"
    assert exc_info.value.errors()[0]["loc"] == ("count",)


async def test_execute_tool_can_skip_validation() -> None:
    @tool()
    def echo(value: str) -> str:
        return value

    assert await execute_tool(echo, {"value": 123}, validate=False) == 123


def test_validate_tool_arguments_for_external_schema_returns_arguments_unchanged() -> None:
    external_tool = Tool(name="external", description="", parameters={"type": "object"})
    arguments = {"value": "raw"}

    assert validate_tool_arguments(external_tool, arguments) is arguments


def test_tool_from_pydantic_validates_arguments() -> None:
    class SearchParams(BaseModel):
        query: str
        max_results: int = 5

    search = tool_from_pydantic(SearchParams, name="search")

    assert validate_tool_arguments(search, {"query": "python", "max_results": "3"}) == {
        "query": "python",
        "max_results": 3,
    }
