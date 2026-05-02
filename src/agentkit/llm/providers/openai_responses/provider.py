from __future__ import annotations

from collections.abc import AsyncIterator

try:
    from openai import AsyncOpenAI
except ImportError as exc:  # pragma: no cover - depends on optional extra
    from agentkit.exceptions import ProviderDependencyError

    raise ProviderDependencyError(
        "OpenAI Responses support requires the 'openai' extra. "
        "Install it with: pip install 'agentkit[openai]'"
    ) from exc

from ..._hooks import apply_payload_hook, apply_response_hook, check_abort
from ...context import Context
from ...streaming import StreamEvent, StreamResponse
from ...types import Content, Message, Response, Role, StopReason, TextContent, ToolCall, Usage
from .._utils import client_with_retries, parse_json_args
from ..base import ModelOptions, Provider
from ._convert import build_request
from ._parse import map_stop_reason, parse_response

__all__ = ["OpenAIResponsesProvider"]


class OpenAIResponsesProvider(Provider):
    """OpenAI Responses API provider."""

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
        return "openai-responses"

    async def complete(self, context: Context, options: ModelOptions) -> Response:
        request = await apply_payload_hook(build_request(context, options), options)
        client = client_with_retries(self._client, options.max_retries)
        response = await client.responses.create(**request, stream=False)
        await apply_response_hook(response, options)
        return parse_response(response, options.model)

    def stream(self, context: Context, options: ModelOptions) -> StreamResponse:
        return StreamResponse(self._stream_events(context, options), model_ref=options.model_ref)

    async def _stream_events(
        self,
        context: Context,
        options: ModelOptions,
    ) -> AsyncIterator[StreamEvent]:
        request = await apply_payload_hook(build_request(context, options), options)
        client = client_with_retries(self._client, options.max_retries)
        stream = await client.responses.create(**request, stream=True)

        text_chunks: list[str] = []
        tool_acc: dict[str, dict[str, str]] = {}
        tool_calls_done: list[ToolCall] = []
        usage = Usage()
        stop_reason = StopReason.STOP
        model_used = options.model

        yield StreamEvent.start(model_used)

        async for event in stream:
            check_abort(options)
            etype = getattr(event, "type", "")

            if etype == "response.created":
                response = event.response
                model_used = getattr(response, "model", model_used)

            elif etype == "response.output_item.added":
                item = event.item
                if getattr(item, "type", None) == "function_call":
                    item_id = item.id or str(getattr(event, "output_index", ""))
                    tool_acc[item_id] = {
                        "call_id": item.call_id,
                        "name": item.name,
                        "args_buf": "",
                    }
                    yield StreamEvent.toolcall_start(item.call_id, item.name)

            elif etype == "response.text.delta":
                text_chunks.append(event.delta)
                yield StreamEvent.text_delta(event.delta)

            elif etype == "response.output_text.delta":
                text_chunks.append(event.delta)
                yield StreamEvent.text_delta(event.delta)

            elif etype in {"response.text.done", "response.output_text.done"}:
                text = getattr(event, "text", None) or getattr(event, "delta", None)
                if text:
                    yield StreamEvent.text_end(text)

            elif etype == "response.function_call_arguments.delta":
                item_id = event.item_id
                if item_id in tool_acc:
                    tool_acc[item_id]["args_buf"] += event.delta
                    yield StreamEvent.toolcall_delta(tool_acc[item_id]["call_id"], event.delta)

            elif etype == "response.function_call_arguments.done":
                item_id = event.item_id
                if item_id in tool_acc:
                    tc_info = tool_acc[item_id]
                    args = parse_json_args(tc_info["args_buf"])
                    tc = ToolCall(
                        id=tc_info["call_id"],
                        name=tc_info["name"],
                        arguments=args,
                    )
                    tool_calls_done.append(tc)
                    yield StreamEvent.toolcall_end(tc_info["call_id"], tc_info["name"], args)

            elif etype == "response.completed":
                response = event.response
                model_used = getattr(response, "model", model_used)
                usage_data = getattr(response, "usage", None)
                if usage_data:
                    details = getattr(usage_data, "input_tokens_details", None)
                    usage = Usage(
                        input=getattr(usage_data, "input", 0)
                        or getattr(usage_data, "input_tokens", 0),
                        output=getattr(usage_data, "output", 0)
                        or getattr(usage_data, "output_tokens", 0),
                        cache_read=getattr(details, "cached_tokens", 0) if details else 0,
                    )
                stop_reason = map_stop_reason(getattr(response, "status", None))

                content: list[Content] = []
                if text_chunks:
                    content.append(TextContent(text="".join(text_chunks)))
                content.extend(tool_calls_done)

                final_response = Response(
                    message=Message(role=Role.ASSISTANT, content=content),
                    stop_reason=stop_reason,
                    usage=usage,
                    model=model_used,
                    raw=response,
                )
                yield StreamEvent.done(final_response)
                break

            elif etype == "response.failed":
                error_info = getattr(event.response, "error", None)
                msg = str(error_info) if error_info else "Response failed"
                yield StreamEvent.error(RuntimeError(msg))
                break
