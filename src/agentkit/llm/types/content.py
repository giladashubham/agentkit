from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["TextContent", "ImageContent", "ToolCall", "ToolResult", "ThinkingContent", "Content"]


class TextContent(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["text"] = "text"
    text: str


class ImageContent(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)
    type: Literal["image"] = "image"
    data: str
    mime_type: str = Field(alias="mimeType", serialization_alias="mimeType")


class ToolCall(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["toolCall"] = "toolCall"
    id: str
    name: str
    arguments: dict[str, Any]


class ToolResult(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)
    type: Literal["toolResult"] = "toolResult"
    tool_call_id: str = Field(alias="toolCallId", serialization_alias="toolCallId")
    tool_name: str = Field(alias="toolName", serialization_alias="toolName")
    content: str | list[TextContent | ImageContent]
    is_error: bool = Field(default=False, alias="isError", serialization_alias="isError")


class ThinkingContent(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["thinking"] = "thinking"
    text: str
    signature: str | None = None
    redacted: bool = False


Content = Annotated[
    TextContent | ImageContent | ToolCall | ToolResult | ThinkingContent,
    Field(discriminator="type"),
]
