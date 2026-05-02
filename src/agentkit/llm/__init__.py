"""Unified LLM provider abstraction with streaming and tool support."""

from .api import complete, get_provider, list_provider_apis, register_provider, stream
from .context import Context
from .model import Model, RunOptions
from .models.presets import OPENAI_COMPATIBLE_BASE_URLS, openai_compatible_model
from .models.registry import get_model, get_models, list_model_providers, register_model
from .providers import ModelOptions, Provider
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
    "list_model_providers",
    "register_model",
    "stream",
    "get_provider",
    "list_provider_apis",
    "register_provider",
    "OPENAI_COMPATIBLE_BASE_URLS",
    "openai_compatible_model",
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
]
