from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, get_type_hints

from pydantic import BaseModel, ValidationError, create_model

from agentkit.exceptions import ToolValidationError

__all__ = ["Tool", "tool", "tool_from_pydantic", "validate_tool_arguments", "execute_tool"]


@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    func: Callable[..., Any] | Callable[..., Awaitable[Any]] | None = None
    args_model: type[BaseModel] | None = field(default=None, repr=False, compare=False)

    def to_anthropic(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def tool(
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable[..., Any]], Tool]:
    """Decorator to create a Tool from a function with type hints."""

    def decorator(func: Callable[..., Any]) -> Tool:
        func_name = name or func.__name__
        func_description = description or func.__doc__ or ""

        sig = inspect.signature(func)
        hints = get_type_hints(func)

        fields: dict[str, Any] = {}
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            annotation = hints.get(param_name, str)
            default = ... if param.default is inspect.Parameter.empty else param.default

            fields[param_name] = (annotation, default)

        model = create_model(f"{func_name}_params", **fields)
        schema = model.model_json_schema()
        schema.pop("title", None)

        return Tool(
            name=func_name,
            description=func_description.strip(),
            parameters=schema,
            func=func,
            args_model=model,
        )

    return decorator


def tool_from_pydantic(
    model: type[BaseModel],
    name: str | None = None,
    description: str | None = None,
    func: Callable[..., Any] | None = None,
) -> Tool:
    """Create a Tool from a Pydantic model."""
    schema = model.model_json_schema()
    schema.pop("title", None)

    return Tool(
        name=name or model.__name__,
        description=description or model.__doc__ or "",
        parameters=schema,
        func=func,
        args_model=model,
    )


def validate_tool_arguments(tool: Tool, arguments: dict[str, Any]) -> dict[str, Any]:
    """Validate and coerce tool-call arguments using the tool's Pydantic model.

    Tools created from raw JSON schema without an associated Pydantic model return
    arguments unchanged because runtime validation is not available.
    """
    if tool.args_model is None:
        return arguments
    try:
        validated = tool.args_model.model_validate(arguments)
    except ValidationError as exc:
        raise ToolValidationError(tool.name, exc) from exc
    return validated.model_dump()


async def execute_tool(tool: Tool, arguments: dict[str, Any], *, validate: bool = True) -> Any:
    """Execute a tool with the given arguments.

    By default, arguments are validated and coerced for tools created with
    `@tool()` or `tool_from_pydantic()`. Pass `validate=False` to execute raw
    arguments directly.
    """
    if tool.func is None:
        raise ValueError(f"Tool {tool.name} has no callable function")

    call_arguments = validate_tool_arguments(tool, arguments) if validate else arguments
    if inspect.iscoroutinefunction(tool.func):
        return await tool.func(**call_arguments)
    return tool.func(**call_arguments)
