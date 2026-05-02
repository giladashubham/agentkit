"""Unified LLM provider abstraction with streaming and tool support."""

from .api import complete, get_provider, list_providers, register_provider, stream
from .context import Context
from .model import Model, RunOptions
from .providers import ModelOptions, Provider
from .registry import get_model, get_models, get_providers, register_model
from .streaming import EventType, StreamEvent, StreamResponse
from .tools import Tool, execute_tool, tool, tool_from_pydantic
from .types import (
    AssistantMessage,
    Content,
    Cost,
    ImageContent,
    Message,
    Response,
    Role,
    StopReason,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResult,
    ToolResultMessage,
    Usage,
    UserMessage,
)

__version__ = "0.1.0"

__all__ = [
    "Model",
    "RunOptions",
    "complete",
    "get_model",
    "get_models",
    "get_providers",
    "register_model",
    "stream",
    "get_provider",
    "list_providers",
    "register_provider",
    "AssistantMessage",
    "Cost",
    "Content",
    "ImageContent",
    "Message",
    "Response",
    "Role",
    "StopReason",
    "TextContent",
    "ThinkingContent",
    "ToolCall",
    "ToolResult",
    "ToolResultMessage",
    "Usage",
    "UserMessage",
    "Context",
    "Tool",
    "tool",
    "tool_from_pydantic",
    "execute_tool",
    "EventType",
    "StreamEvent",
    "StreamResponse",
    "ModelOptions",
    "Provider",
    "AnthropicProvider",
    "OpenAIProvider",
    "OpenAIWebSocketSession",
]


def __getattr__(name: str):
    provider_exports = {
        "AnthropicProvider",
        "OpenAIProvider",
        "OpenAIWebSocketSession",
    }
    if name in provider_exports:
        from . import providers

        return getattr(providers, name)
    raise AttributeError(name)
