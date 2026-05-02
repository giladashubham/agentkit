"""Unified LLM provider abstraction with streaming and tool support."""

from agentkit import __version__

from .api import complete, get_provider, list_provider_apis, register_provider, stream
from .context import Context
from .env import get_env_api_key
from .model import Model, ModelCost, RunOptions
from .models.costs import calculate_cost
from .models.presets import OPENAI_COMPATIBLE_BASE_URLS, openai_compatible_model
from .models.registry import get_model, get_models, list_model_providers, register_model
from .providers import ModelOptions, Provider
from .streaming import EventType, StreamEvent, StreamResponse
from .tools import Tool, execute_tool, tool, tool_from_pydantic, validate_tool_arguments
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

__all__ = [
    "__version__",
    "Model",
    "ModelCost",
    "RunOptions",
    "complete",
    "get_model",
    "get_models",
    "list_model_providers",
    "register_model",
    "calculate_cost",
    "stream",
    "get_provider",
    "list_provider_apis",
    "register_provider",
    "get_env_api_key",
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
    "validate_tool_arguments",
    "execute_tool",
    "EventType",
    "StreamEvent",
    "StreamResponse",
    "ModelOptions",
    "Provider",
]
