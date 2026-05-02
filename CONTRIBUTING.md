# Contributing

Thanks for helping improve AgentKit.

## Development setup

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
uv build
```

Provider SDKs are optional. Install extras only when working on that provider:

```bash
uv sync --extra dev --extra openai
uv sync --extra dev --extra anthropic
uv sync --extra dev --extra google
```

## Guidelines

- Keep the public API small and explicit.
- Avoid live provider API calls in tests.
- Put provider-specific code under `src/agentkit/llm/providers/<provider>/`.
- Add tests for converter/parser behavior when changing provider integrations.
- Prefer small, focused pull requests.
