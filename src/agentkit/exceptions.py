"""AgentKit exceptions."""


class AgentKitError(Exception):
    """Base class for AgentKit errors."""


class ProviderDependencyError(AgentKitError, ImportError):
    """Raised when an optional provider dependency is not installed."""
