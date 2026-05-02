from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

try:
    from openai import AsyncOpenAI
except ImportError as exc:  # pragma: no cover - depends on optional extra
    from agentkit.exceptions import ProviderDependencyError

    raise ProviderDependencyError(
        "OpenAI support requires the 'openai' extra. "
        "Install it with: pip install 'agentkit[openai]'"
    ) from exc

from ..._hooks import apply_payload_hook, apply_response_hook, check_abort
from ...context import Context
from ...streaming import StreamEvent, StreamResponse
from ...types import Content, Message, Response, Role, StopReason, TextContent, ToolCall, Usage
from .._utils import client_with_retries, safe_json_loads
from ..base import ModelOptions, Provider
from ._convert import build_request
from ._parse import map_stop_reason, parse_response

__all__ = ["OpenAIProvider"]


class OpenAIProvider(Provider):
    """OpenAI provider (also works with compatible APIs)."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
    ):
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
        )

    @property
    def name(self) -> str:
        return "openai"

    async def complete(self, context: Context, options: ModelOptions) -> Response:
        request = await apply_payload_hook(build_request(context, options), options)
        client = client_with_retries(self._client, options.max_retries)
        response = await client.chat.completions.create(**request, stream=False)
        await apply_response_hook(response, options)
        return parse_response(response, options.model)

    def stream(self, context: Context, options: ModelOptions) -> StreamResponse:
        return StreamResponse(self._stream_events(context, options), model_ref=options.model_ref)

    async def _stream_events(
        self,
        context: Context,
        options: ModelOptions,
    ) -> AsyncIterator[StreamEvent]:
        accumulated_text = ""
        tool_calls: dict[int, dict[str, Any]] = {}
        usage = Usage()
        stop_reason = StopReason.STOP
        model = options.model

        request = await apply_payload_hook(build_request(context, options), options)
        client = client_with_retries(self._client, options.max_retries)
        stream = await client.chat.completions.create(
            **request,
            stream=True,
            stream_options={"include_usage": True},
        )

        yield StreamEvent.start(model)

        async for chunk in stream:
            check_abort(options)
            if chunk.usage:
                usage.input = chunk.usage.prompt_tokens
                usage.output = chunk.usage.completion_tokens

            if not chunk.choices:
                continue

            choice = chunk.choices[0]

            if choice.finish_reason:
                stop_reason = map_stop_reason(choice.finish_reason)

            delta = choice.delta
            if delta is None:
                continue

            if delta.content:
                accumulated_text += delta.content
                yield StreamEvent.text_delta(delta.content)

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls:
                        tool_calls[idx] = {
                            "id": tc.id or "",
                            "name": tc.function.name if tc.function else "",
                            "arguments_json": "",
                        }
                        if tc.id and tc.function and tc.function.name:
                            yield StreamEvent.toolcall_start(tc.id, tc.function.name)

                    if tc.function and tc.function.arguments:
                        tool_calls[idx]["arguments_json"] += tc.function.arguments
                        yield StreamEvent.toolcall_delta(
                            tool_calls[idx]["id"],
                            tc.function.arguments,
                        )

        if accumulated_text:
            yield StreamEvent.text_end(accumulated_text)

        for tc in tool_calls.values():
            yield StreamEvent.toolcall_end(
                tc["id"],
                tc["name"],
                safe_json_loads(tc["arguments_json"]),
            )

        content: list[Content] = []
        if accumulated_text:
            content.append(TextContent(text=accumulated_text))
        for tc in tool_calls.values():
            content.append(ToolCall(
                id=tc["id"],
                name=tc["name"],
                arguments=safe_json_loads(tc["arguments_json"]),
            ))

        response = Response(
            message=Message(role=Role.ASSISTANT, content=content),
            stop_reason=stop_reason,
            usage=usage,
            model=model,
        )
        yield StreamEvent.done(response)
