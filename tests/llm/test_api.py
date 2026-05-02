from __future__ import annotations

from agentkit.llm import Context, Model, RunOptions, tool


def test_model_carries_provider_api_and_id() -> None:
    model = Model(
        provider="openai",
        api="openai-completions",
        id="gpt-4o-mini",
        base_url="https://example.com",
    )

    assert model.provider == "openai"
    assert model.api == "openai-completions"
    assert model.id == "gpt-4o-mini"
    assert model.base_url == "https://example.com"


def test_run_options_are_request_scoped() -> None:
    options = RunOptions(
        max_tokens=100,
        temperature=0.2,
        tool_choice="auto",
        reasoning="high",
        reasoning_budget=8000,
        timeout_ms=30000,
        max_retries=2,
        headers={"x-test": "true"},
    )

    assert options.max_tokens == 100
    assert options.temperature == 0.2
    assert options.tool_choice == "auto"
    assert options.reasoning == "high"
    assert options.reasoning_budget == 8000
    assert options.timeout_ms == 30000
    assert options.max_retries == 2
    assert options.headers == {"x-test": "true"}


def test_context_can_own_tools_for_agent_loop() -> None:
    @tool()
    def search(query: str) -> str:
        """Search."""
        return query

    ctx = Context(system_prompt="You are helpful.", tools=[search])
    ctx.add_user("Find docs")

    data = ctx.to_dict()
    restored = Context.from_dict(data)

    assert data["tools"][0]["name"] == "search"
    assert restored.tools[0].name == "search"
    assert restored.tools[0].func is None
