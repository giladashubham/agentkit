from __future__ import annotations

import asyncio

import pytest

from agentkit.llm._hooks import apply_payload_hook, check_abort
from agentkit.llm.providers.base import ModelOptions


async def test_payload_hook_can_replace_payload() -> None:
    options = ModelOptions(
        model="test",
        on_payload=lambda payload, model: {**payload, "temperature": 0.1},
    )

    payload = await apply_payload_hook({"model": "test"}, options)

    assert payload == {"model": "test", "temperature": 0.1}


def test_abort_signal_is_explicit() -> None:
    signal = asyncio.Event()
    signal.set()
    options = ModelOptions(model="test", abort_signal=signal)

    with pytest.raises(RuntimeError, match="aborted"):
        check_abort(options)
