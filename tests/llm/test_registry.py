from __future__ import annotations

import pytest

from agentkit.llm import (
    AssistantMessage,
    Context,
    Model,
    ModelCost,
    Response,
    RunOptions,
    StopReason,
    TextContent,
    Usage,
    complete,
    list_provider_apis,
)
from agentkit.llm.providers.base import ModelOptions, Provider
from agentkit.llm.providers.builtins import register_builtin_providers
from agentkit.llm.providers.registry import clear_providers, register_provider
from agentkit.llm.streaming import StreamResponse


class DummyProvider(Provider):
    @property
    def name(self) -> str:
        return "dummy"

    async def complete(self, context: Context, options: ModelOptions) -> Response:
        return Response(
            message=context.messages[-1],
            stop_reason=StopReason.STOP,
            usage=Usage(input=100, output=50),
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
    assert list_provider_apis() == ["dummy"]


async def test_model_defaults_flow_into_run_options() -> None:
    clear_providers()
    captured: list[ModelOptions] = []

    class CapturingProvider(DummyProvider):
        async def complete(self, context: Context, options: ModelOptions) -> Response:
            captured.append(options)
            return await super().complete(context, options)

    register_provider("dummy", lambda model: CapturingProvider())

    model = Model(
        provider="dummy",
        api="dummy",
        id="dummy-model",
        max_tokens=123,
        headers={"x-model": "yes", "x-shared": "model"},
    )
    ctx = Context().add_user("hello")

    await complete(model, ctx)

    assert captured[0].max_tokens == 123
    assert captured[0].headers == {"x-model": "yes", "x-shared": "model"}


async def test_run_options_override_model_defaults() -> None:
    clear_providers()
    captured: list[ModelOptions] = []

    class CapturingProvider(DummyProvider):
        async def complete(self, context: Context, options: ModelOptions) -> Response:
            captured.append(options)
            return await super().complete(context, options)

    register_provider("dummy", lambda model: CapturingProvider())

    model = Model(
        provider="dummy",
        api="dummy",
        id="dummy-model",
        max_tokens=123,
        headers={"x-model": "yes", "x-shared": "model"},
    )
    ctx = Context().add_user("hello")

    await complete(
        model,
        ctx,
        RunOptions(max_tokens=456, headers={"x-run": "yes", "x-shared": "run"}),
    )

    assert captured[0].max_tokens == 456
    assert captured[0].headers == {"x-model": "yes", "x-run": "yes", "x-shared": "run"}


async def test_complete_attaches_assistant_response_metadata() -> None:
    clear_providers()

    class AssistantProvider(DummyProvider):
        async def complete(self, context: Context, options: ModelOptions) -> Response:
            return Response(
                message=AssistantMessage(content=[TextContent(text="hello")]),
                stop_reason=StopReason.STOP,
                usage=Usage(input=100, output=50),
                model="actual-model",
            )

    register_provider("dummy", lambda model: AssistantProvider())
    model = Model(provider="dummy-provider", api="dummy", id="requested-model")

    response = await complete(model, Context().add_user("hello"))

    assert response.message.provider == "dummy-provider"
    assert response.message.api == "dummy"
    assert response.message.model == "requested-model"
    assert response.message.response_model == "actual-model"
    assert response.message.usage == response.usage.model_dump(mode="json")
    assert response.message.stop_reason == "stop"


async def test_complete_calculates_response_cost() -> None:
    clear_providers()
    register_provider("dummy", lambda model: DummyProvider())

    model = Model(
        provider="dummy",
        api="dummy",
        id="dummy-model",
        cost=ModelCost(input=1.00, output=2.00),
    )
    ctx = Context().add_user("hello")

    response = await complete(model, ctx)

    assert response.usage.cost.input == pytest.approx(0.0001)
    assert response.usage.cost.output == pytest.approx(0.0001)
    assert response.usage.cost.total == pytest.approx(0.0002)


def test_unknown_provider_lists_available_providers() -> None:
    clear_providers()
    register_provider("dummy", lambda model: DummyProvider())

    with pytest.raises(ValueError, match="Unknown api: missing. Available: dummy"):
        from agentkit.llm import get_provider

        get_provider(Model(provider="missing", api="missing", id="x"))
