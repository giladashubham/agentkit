from __future__ import annotations

import inspect
from typing import Any

from .providers.base import ModelOptions
from .types import StopReason


def check_abort(options: ModelOptions) -> None:
    if options.abort_signal is not None and options.abort_signal.is_set():
        raise RuntimeError(StopReason.ABORTED.value)


async def apply_payload_hook(payload: dict[str, Any], options: ModelOptions) -> dict[str, Any]:
    check_abort(options)
    if options.on_payload is None:
        return payload
    result = options.on_payload(payload, options.model_ref)
    if inspect.isawaitable(result):
        result = await result
    check_abort(options)
    return payload if result is None else result


async def apply_response_hook(response: Any, options: ModelOptions) -> None:
    check_abort(options)
    if options.on_response is None:
        return
    result = options.on_response(response, options.model_ref)
    if inspect.isawaitable(result):
        await result
    check_abort(options)
