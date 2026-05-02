from __future__ import annotations

from collections.abc import AsyncIterator

try:
    from google import genai
    from google.genai import types
except ImportError as exc:  # pragma: no cover - depends on optional extra
    from agentkit.exceptions import ProviderDependencyError

    raise ProviderDependencyError(
        "Google support requires the 'google' extra. "
        "Install it with: pip install 'agentkit[google]'"
    ) from exc

from ..._hooks import apply_payload_hook, apply_response_hook, check_abort
from ...context import Context
from ...streaming import StreamEvent, StreamResponse
from ...types import Content, Message, Response, Role, StopReason, TextContent, Usage
from .._utils import client_with_retries
from ..base import ModelOptions, Provider
from ._convert import build_request
from ._parse import map_stop_reason, parse_response

__all__ = ["GoogleProvider"]


class GoogleProvider(Provider):
    """Google Gemini provider for Developer API and Vertex AI."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        vertexai: bool = False,
        project: str | None = None,
        location: str | None = None,
    ):
        client_kwargs = {
            "api_key": api_key,
            "vertexai": vertexai,
            "project": project,
            "location": location,
        }
        if base_url is not None:
            client_kwargs["http_options"] = types.HttpOptions(base_url=base_url)
        self._client = genai.Client(**{k: v for k, v in client_kwargs.items() if v is not None}).aio
        self._name = "google-vertex" if vertexai else "google"

    @property
    def name(self) -> str:
        return self._name

    async def complete(self, context: Context, options: ModelOptions) -> Response:
        request = await apply_payload_hook(build_request(context, options, types), options)
        client = client_with_retries(self._client, options.max_retries)
        response = await client.models.generate_content(**request)
        await apply_response_hook(response, options)
        return parse_response(response, options.model)

    def stream(self, context: Context, options: ModelOptions) -> StreamResponse:
        return StreamResponse(self._stream_events(context, options))

    async def _stream_events(
        self,
        context: Context,
        options: ModelOptions,
    ) -> AsyncIterator[StreamEvent]:
        request = await apply_payload_hook(build_request(context, options, types), options)
        client = client_with_retries(self._client, options.max_retries)
        stream = await client.models.generate_content_stream(**request)

        text_chunks: list[str] = []
        usage = Usage()
        stop_reason = StopReason.STOP

        yield StreamEvent.start(options.model)

        async for chunk in stream:
            check_abort(options)
            text = getattr(chunk, "text", None)
            if text:
                text_chunks.append(text)
                yield StreamEvent.text_delta(text)

            usage_metadata = getattr(chunk, "usage_metadata", None)
            if usage_metadata:
                usage.input = getattr(usage_metadata, "prompt_token_count", 0) or usage.input
                usage.output = getattr(usage_metadata, "candidates_token_count", 0) or usage.output

            candidates = getattr(chunk, "candidates", None) or []
            if candidates:
                stop_reason = map_stop_reason(getattr(candidates[0], "finish_reason", None))

        text = "".join(text_chunks)
        if text:
            yield StreamEvent.text_end(text)

        content: list[Content] = [TextContent(text=text)] if text else []
        response = Response(
            message=Message(role=Role.ASSISTANT, content=content),
            stop_reason=stop_reason,
            usage=usage,
            model=options.model,
        )
        yield StreamEvent.done(response)
