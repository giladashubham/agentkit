from __future__ import annotations

from enum import StrEnum
from time import time
from typing import Literal

from pydantic import BaseModel, Field

from .content import Content, ImageContent, TextContent, ThinkingContent, ToolCall, ToolResult

__all__ = ["Role", "Message", "UserMessage", "AssistantMessage", "ToolResultMessage"]


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_RESULT = "toolResult"
    SYSTEM = "system"


class Message(BaseModel):
    role: Role
    content: list[Content]
    timestamp: float = Field(default_factory=time)

    @classmethod
    def user(cls, text: str) -> UserMessage:
        return UserMessage(content=[TextContent(text=text)])

    @classmethod
    def assistant(cls, text: str) -> AssistantMessage:
        return AssistantMessage(content=[TextContent(text=text)])

    @classmethod
    def assistant_with_tool_calls(
        cls,
        tool_calls: list[ToolCall],
        text: str | None = None,
        thinking: str | None = None,
    ) -> AssistantMessage:
        content: list[Content] = []
        if thinking:
            content.append(ThinkingContent(text=thinking))
        if text:
            content.append(TextContent(text=text))
        content.extend(tool_calls)
        return AssistantMessage(content=content)

    @classmethod
    def tool_result(
        cls,
        tool_call_id: str,
        tool_name: str,
        content: str | list[TextContent | ImageContent],
        is_error: bool = False,
    ) -> ToolResultMessage:
        return ToolResultMessage(
            content=[
                ToolResult(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    content=content,
                    is_error=is_error,
                )
            ]
        )

    def text(self) -> str:
        return "".join(c.text for c in self.content if isinstance(c, TextContent))

    def tool_calls(self) -> list[ToolCall]:
        return [c for c in self.content if isinstance(c, ToolCall)]

    def thinking(self) -> str:
        return "".join(c.text for c in self.content if isinstance(c, ThinkingContent))


class UserMessage(Message):
    role: Literal[Role.USER] = Role.USER
    content: list[TextContent | ImageContent]


class AssistantMessage(Message):
    role: Literal[Role.ASSISTANT] = Role.ASSISTANT
    content: list[TextContent | ThinkingContent | ToolCall]


class ToolResultMessage(Message):
    role: Literal[Role.TOOL_RESULT] = Role.TOOL_RESULT
    content: list[ToolResult]
