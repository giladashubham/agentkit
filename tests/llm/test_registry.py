from __future__ import annotations

import pytest

from agentkit.llm import Context, Model, Response, StopReason, Usage, complete, list_providers
from agentkit.llm.providers.base import ModelOptions, Provider
from agentkit.llm.providers.registry import (
    clear_providers,
    register_builtin_providers,
    register_provider,
)
from agentkit.llm.streaming import StreamResponse


class DummyProvider(Provider):
    @property
    def name(self) -> str:
        return "dummy"

    async def complete(self, context: Context, options: ModelOptions) -> Response:
        return Response(
            message=context.messages[-1],
            stop_reason=StopReason.STOP,
            usage=Usage(),
            model=options.model,
        )

    def stream(self, context: Context, options: ModelOptions) -> StreamResponse:
        raise NotImplementedError

    async def _stream_events(self, context: Context, options: ModelOptions):
        raise NotImplementedError


@pytest.fixture(autouse=True)
def restore_builtin_providers():
    yield
    clear_providers()
    register_builtin_providers()


async def test_register_custom_provider_factory() -> None:
    clear_providers()
    register_provider("dummy", lambda model: DummyProvider())

    ctx = Context().add_user("hello")
    response = await complete(Model(provider="dummy", api="dummy", id="dummy-model"), ctx)

    assert response.text() == "hello"
    assert response.model == "dummy-model"
    assert list_providers() == ["dummy"]


def test_unknown_provider_lists_available_providers() -> None:
    clear_providers()
    register_provider("dummy", lambda model: DummyProvider())

    with pytest.raises(ValueError, match="Unknown api: missing. Available: dummy"):
        from agentkit.llm import get_provider

        get_provider(Model(provider="missing", api="missing", id="x"))
