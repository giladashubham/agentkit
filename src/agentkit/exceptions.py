"""AgentKit exceptions."""

from typing import Any

from pydantic import ValidationError


class AgentKitError(Exception):
    """Base class for AgentKit errors."""


class ProviderDependencyError(AgentKitError, ImportError):
    """Raised when an optional provider dependency is not installed."""


class ToolValidationError(AgentKitError, ValueError):
    """Raised when tool-call arguments fail validation."""

    def __init__(self, tool_name: str, error: ValidationError):
        self.tool_name = tool_name
        self.validation_error = error
        super().__init__(f"Invalid arguments for tool {tool_name}: {error}")

    def errors(self) -> list[dict[str, Any]]:
        """Return Pydantic validation error details."""
        return self.validation_error.errors()
