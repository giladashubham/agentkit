from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

try:
    from openai import AsyncOpenAI
except ImportError as exc:  # pragma: no cover - depends on optional extra
    from agentkit.exceptions import ProviderDependencyError

    raise ProviderDependencyError(
        "OpenAI WebSocket support requires the 'openai' extra. "
        "Install it with: pip install 'agentkit[openai]'"
    ) from exc

from ..._hooks import apply_payload_hook, check_abort
from ...context import Context
from ...streaming import StreamEvent, StreamResponse
from ...tools import Tool
from ...types import Content, Message, Response, Role, StopReason, TextContent, ToolCall, Usage
from ..base import ModelOptions
from ._convert import convert_context_to_input, convert_tool_choice, tool_to_responses_format

__all__ = ["OpenAIWebSocketSession"]


class OpenAIWebSocketSession:
    """Persistent WebSocket session to the OpenAI Responses API.

    Use as an async context manager — opens the connection on enter, closes on exit.
    Reuse a single instance across many agent turns.

        async with OpenAIWebSocketSession(client, options) as session:
            stream = session.send(ctx)
            async for event in stream:
                ...
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        options: ModelOptions,
        system: str | None = None,
    ) -> None:
        self._client = client
        self._options = options
        self._system = system
        self._conn: Any = None
        self._manager: Any = None
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> OpenAIWebSocketSession:
        try:
            self._manager = self._client.responses.connect()
            self._conn = await self._manager.__aenter__()
        except ImportError:
            raise ImportError(
                "WebSocket support requires the 'websockets' package. "
                "Install it with: pip install 'openai[realtime]>=2.33.0'"
            ) from None
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._manager is not None:
            await self._manager.__aexit__(*exc)

    def send(
        self,
        context: Context,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
    ) -> StreamResponse:
        """Send one agent turn and return a StreamResponse.

        Call `await stream.result()` or iterate the stream before calling
        `send()` again — the session serialises turns via an internal lock.
        """
        opts = ModelOptions(
            model=model or self._options.model,
            max_tokens=max_tokens if max_tokens is not None else self._options.max_tokens,
            temperature=temperature if temperature is not None else self._options.temperature,
            tools=tools if tools is not None else self._options.tools,
            tool_choice=tool_choice if tool_choice is not None else self._options.tool_choice,
            transport="websocket",
            extra=self._options.extra,
        )
        system = context.system_prompt or self._system
        return StreamResponse(self._stream_one_turn(context, opts, system))

    async def _stream_one_turn(
        self,
        context: Context,
        options: ModelOptions,
        system: str | None,
    ) -> AsyncIterator[StreamEvent]:
        async with self._lock:
            create_kwargs: dict[str, Any] = {
                "model": options.model,
                "input": convert_context_to_input(context),
                "stream": True,
            }
            if system:
                create_kwargs["instructions"] = system
            if options.max_tokens:
                create_kwargs["max_output_tokens"] = options.max_tokens
            if options.temperature is not None:
                create_kwargs["temperature"] = options.temperature
            if options.tools:
                create_kwargs["tools"] = [tool_to_responses_format(t) for t in options.tools]
            if options.timeout_ms is not None:
                create_kwargs["timeout"] = options.timeout_ms
            if options.headers:
                create_kwargs["extra_headers"] = options.headers
            if options.tool_choice:
                tool_choice = convert_tool_choice(options.tool_choice)
                if tool_choice is not None:
                    create_kwargs["tool_choice"] = tool_choice

            create_kwargs = await apply_payload_hook(create_kwargs, options)
            await self._conn.response.create(**create_kwargs)

            tool_acc: dict[str, dict[str, str]] = {}

            text_chunks: list[str] = []
            tool_calls_done: list[ToolCall] = []
            usage = Usage()
            stop_reason = StopReason.STOP
            model_used = options.model

            async for event in self._conn:
                check_abort(options)
                etype = event.type

                if etype == "response.created":
                    model_used = event.response.model
                    yield StreamEvent.start(model_used)

                elif etype == "response.output_item.added":
                    item = event.item
                    if getattr(item, "type", None) == "function_call":
                        item_id = item.id or str(event.output_index)
                        tool_acc[item_id] = {
                            "call_id": item.call_id,
                            "name": item.name,
                            "args_buf": "",
                        }
                        yield StreamEvent.toolcall_start(item.call_id, item.name)

                elif etype == "response.text.delta":
                    text_chunks.append(event.delta)
                    yield StreamEvent.text_delta(event.delta)

                elif etype == "response.text.done":
                    yield StreamEvent.text_end(event.text)

                elif etype == "response.function_call_arguments.delta":
                    item_id = event.item_id
                    if item_id in tool_acc:
                        tool_acc[item_id]["args_buf"] += event.delta
                        yield StreamEvent.toolcall_delta(tool_acc[item_id]["call_id"], event.delta)

                elif etype == "response.function_call_arguments.done":
                    item_id = event.item_id
                    if item_id in tool_acc:
                        tc_info = tool_acc[item_id]
                        try:
                            arguments = (
                                json.loads(tc_info["args_buf"]) if tc_info["args_buf"] else {}
                            )
                        except json.JSONDecodeError:
                            arguments = {}
                        tc = ToolCall(
                            id=tc_info["call_id"],
                            name=tc_info["name"],
                            arguments=arguments,
                        )
                        tool_calls_done.append(tc)
                        yield StreamEvent.toolcall_end(
                            tc_info["call_id"],
                            tc_info["name"],
                            arguments,
                        )

                elif etype == "response.completed":
                    resp = event.response
                    if hasattr(resp, "usage") and resp.usage:
                        cached = 0
                        details = getattr(resp.usage, "input_tokens_details", None)
                        if details:
                            cached = getattr(details, "cached_tokens", 0) or 0
                        usage = Usage(
                            input=resp.usage.input,
                            output=resp.usage.output,
                            cache_read=cached,
                        )
                    status = getattr(resp, "status", "completed")
                    stop_reason = StopReason.LENGTH if status == "incomplete" else StopReason.STOP

                    content: list[Content] = []
                    if text_chunks:
                        content.append(TextContent(text="".join(text_chunks)))
                    content.extend(tool_calls_done)

                    final_response = Response(
                        message=Message(role=Role.ASSISTANT, content=content),
                        stop_reason=stop_reason,
                        usage=usage,
                        model=model_used,
                    )
                    yield StreamEvent.done(final_response)
                    break

                elif etype == "response.failed":
                    error_info = getattr(event.response, "error", None)
                    msg = str(error_info) if error_info else "Response failed"
                    yield StreamEvent.error(RuntimeError(msg))
                    break
