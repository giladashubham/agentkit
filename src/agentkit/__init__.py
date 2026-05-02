"""AgentKit package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("agentkit")
except PackageNotFoundError:  # pragma: no cover - source tree fallback
    __version__ = "0.0.0"

__all__ = ["__version__"]
