from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import TypeAdapter

from .tools import Tool
from .types import (
    AssistantMessage,
    Content,
    ImageContent,
    Message,
    Role,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResult,
    ToolResultMessage,
    UserMessage,
)

__all__ = ["Context"]

_content_adapter: TypeAdapter[Content] = TypeAdapter(Content)


@dataclass(slots=True)
class Context:
    """Serializable conversation context."""

    messages: list[Message] = field(default_factory=list)
    system_prompt: str | None = None
    tools: list[Tool] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_user(self, text: str) -> Context:
        self.messages.append(Message.user(text))
        return self

    def add_assistant(self, text: str) -> Context:
        self.messages.append(Message.assistant(text))
        return self

    def add_message(self, message: Message) -> Context:
        self.messages.append(message)
        return self

    def add_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        content: str | list[TextContent | ImageContent],
        is_error: bool = False,
    ) -> Context:
        self.messages.append(Message.tool_result(tool_call_id, tool_name, content, is_error))
        return self

    def clear(self) -> Context:
        self.messages.clear()
        return self

    def copy(self) -> Context:
        return Context(
            messages=[m.model_copy(deep=True) for m in self.messages],
            system_prompt=self.system_prompt,
            tools=list(self.tools),
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a Pi-style JSON-compatible dict."""
        return {
            "systemPrompt": self.system_prompt,
            "messages": [_message_to_dict(m) for m in self.messages],
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
                for tool in self.tools
            ],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Context:
        """Deserialize from a Pi-style dict."""
        return cls(
            messages=[_message_from_dict(m) for m in data.get("messages", [])],
            system_prompt=data.get("systemPrompt"),
            tools=[
                Tool(
                    name=t["name"],
                    description=t.get("description", ""),
                    parameters=t.get("parameters", {"type": "object"}),
                )
                for t in data.get("tools", [])
            ],
            metadata=data.get("metadata", {}),
        )


def _message_to_dict(message: Message) -> dict[str, Any]:
    if message.role == Role.TOOL_RESULT:
        result = next(c for c in message.content if isinstance(c, ToolResult))
        content = result.content
        if isinstance(content, str):
            content = [TextContent(text=content)]
        return {
            "role": message.role.value,
            "toolCallId": result.tool_call_id,
            "toolName": result.tool_name,
            "content": [_content_to_dict(c) for c in content],
            "isError": result.is_error,
            "timestamp": message.timestamp,
        }

    text = message.text()
    serialized_content: str | list[dict[str, Any]]
    if message.role == Role.USER and len(message.content) == 1 and text:
        serialized_content = text
    else:
        serialized_content = [_content_to_dict(c) for c in message.content]

    return {
        "role": message.role.value,
        "content": serialized_content,
        "timestamp": message.timestamp,
    }


def _message_from_dict(data: dict[str, Any]) -> Message:
    role = Role(data["role"])
    timestamp = data.get("timestamp", 0)

    if role == Role.USER:
        content = data.get("content", "")
        if isinstance(content, str):
            return UserMessage(content=[TextContent(text=content)], timestamp=timestamp)
        user_content = [
            c
            for c in (_content_from_dict(item) for item in content)
            if isinstance(c, (TextContent, ImageContent))
        ]
        return UserMessage(content=user_content, timestamp=timestamp)

    if role == Role.ASSISTANT:
        content = data.get("content", [])
        if isinstance(content, str):
            return AssistantMessage(content=[TextContent(text=content)], timestamp=timestamp)
        return AssistantMessage(
            content=[
                c
                for c in (_content_from_dict(item) for item in content)
                if isinstance(c, (TextContent, ThinkingContent, ToolCall))
            ],
            timestamp=timestamp,
        )

    if role == Role.TOOL_RESULT:
        content = [_content_from_dict(item) for item in data.get("content", [])]
        return ToolResultMessage(
            content=[
                ToolResult(
                    tool_call_id=data["toolCallId"],
                    tool_name=data["toolName"],
                    content=[c for c in content if isinstance(c, (TextContent, ImageContent))],
                    is_error=data.get("isError", False),
                )
            ],
            timestamp=timestamp,
        )

    raise ValueError(f"Unknown message role: {role}")


def _content_to_dict(content: Content) -> dict[str, Any]:
    return content.model_dump(mode="json", by_alias=True)


def _content_from_dict(data: dict[str, Any]) -> Content:
    if data.get("type") == "thinking" and "thinking" in data:
        data = {**data, "text": data["thinking"]}
    return _content_adapter.validate_python(data)
