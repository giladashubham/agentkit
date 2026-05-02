from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from .content import ToolCall
from .messages import Message

__all__ = ["StopReason", "Cost", "Usage", "Response"]


class StopReason(StrEnum):
    STOP = "stop"
    TOOL_USE = "toolUse"
    LENGTH = "length"
    ABORTED = "aborted"
    ERROR = "error"


class Cost(BaseModel):
    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    total: float = 0.0


class Usage(BaseModel):
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cost: Cost = Field(default_factory=Cost)

    @property
    def total_tokens(self) -> int:
        return self.input + self.output


class Response(BaseModel):
    message: Message
    stop_reason: StopReason
    usage: Usage
    model: str
    raw: Any = Field(default=None, repr=False, exclude=True)

    def text(self) -> str:
        return self.message.text()

    def tool_calls(self) -> list[ToolCall]:
        return self.message.tool_calls()

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls()) > 0
