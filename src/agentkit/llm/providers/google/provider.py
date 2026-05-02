from __future__ import annotations

from ...context import Context
from ...streaming import StreamResponse
from ...types import Response
from ..base import ModelOptions, Provider

__all__ = ["GoogleProvider"]


class GoogleProvider(Provider):
    """Google GenAI provider — not yet implemented."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        raise NotImplementedError(
            "GoogleProvider is not yet implemented. "
            "Install the optional dep: pip install 'agentkit[google]'"
        )

    @property
    def name(self) -> str:
        return "google"

    async def complete(self, context: Context, options: ModelOptions) -> Response:
        raise NotImplementedError

    def stream(self, context: Context, options: ModelOptions) -> StreamResponse:
        raise NotImplementedError

    async def _stream_events(self, context: Context, options: ModelOptions):
        raise NotImplementedError
