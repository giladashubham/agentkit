# AgentKit

Simple, extensible building blocks for LLM apps and agent loops.

AgentKit is designed around a **model/context-first** API built for agent loops:

```python
response = await complete(model, context, options)
```

This keeps the future agent loop simple: the loop carries a `Model`, a `Context`, and request-scoped `RunOptions`.

## Installation

Core install:

```bash
pip install agentkit
```

Optional provider extras:

```bash
pip install "agentkit[openai]"
pip install "agentkit[anthropic]"
pip install "agentkit[google]"
pip install "agentkit[all]"
```

With `uv`:

```bash
uv add agentkit
uv add "agentkit[openai]"
```

## Quick start

```python
import asyncio

from agentkit.llm import Context, Model, complete


async def main() -> None:
    model = Model(provider="anthropic", api="anthropic-messages", id="claude-sonnet-4-20250514")
    context = Context(system_prompt="You are a helpful assistant.")
    context.add_user("What is the capital of France?")

    response = await complete(model, context)
    print(response.text())


asyncio.run(main())
```

## Supported providers and models

AgentKit has a small built-in provider registry and a lightweight model registry. It does **not** try to ship a generated database of every model from every vendor.

### Built-in API providers

| API | Provider package | Status | Install extra |
|---|---|---|---|
| `anthropic-messages` | Anthropic | Implemented | `agentkit[anthropic]` |
| `openai-completions` | OpenAI Chat Completions | Implemented | `agentkit[openai]` |
| `openai-responses` | OpenAI Responses API | Implemented | `agentkit[openai]` |
| OpenAI Responses WebSocket | OpenAI persistent session API | Implemented via `openai_ws` | `agentkit[openai]` |
| `google-generative-ai` | Google Gemini Developer API | Implemented | `agentkit[google]` |
| `google-vertex` | Google Gemini on Vertex AI | Implemented | `agentkit[google]` |

### Built-in model registry

| Provider | Model IDs |
|---|---|
| `anthropic` | `claude-sonnet-4-20250514` |
| `openai` | `gpt-4o-mini`, `gpt-4o` |
| `deepseek` | `deepseek-chat` |
| `groq` | `llama-3.3-70b-versatile` |
| `openrouter` | `openai/gpt-4o-mini` |
| `xai` | `grok-4` |
| `fireworks` | `accounts/fireworks/models/llama-v3p1-70b-instruct` |
| `together` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` |
| `ollama` | `llama3.2` |
| `google` | `gemini-2.5-flash`, `gemini-2.5-pro` |
| `google-vertex` | `gemini-2.5-flash`, `gemini-2.5-pro` |

Use `get_model(provider, model_id)` for built-ins, or construct `Model(...)` directly for any model ID supported by the underlying provider.

### OpenAI-compatible preset providers

These providers reuse the OpenAI-compatible implementation with preset `base_url` values:

| Provider | Preset base URL |
|---|---|
| `deepseek` | `https://api.deepseek.com` |
| `groq` | `https://api.groq.com/openai/v1` |
| `openrouter` | `https://openrouter.ai/api/v1` |
| `xai` | `https://api.x.ai/v1` |
| `fireworks` | `https://api.fireworks.ai/inference/v1` |
| `together` | `https://api.together.xyz/v1` |
| `ollama` | `http://localhost:11434/v1` |
| `perplexity` | `https://api.perplexity.ai` |
| `cerebras` | `https://api.cerebras.ai/v1` |
| `sambanova` | `https://api.sambanova.ai/v1` |
| `nebius` | `https://api.studio.nebius.ai/v1` |

## OpenAI

Chat Completions:

```python
from agentkit.llm import Context, Model, complete

model = Model(provider="openai", api="openai-completions", id="gpt-4o-mini")
context = Context()
context.add_user("Say hello in three words.")

response = await complete(model, context)
print(response.text())
```

Responses API:

```python
model = Model(provider="openai", api="openai-responses", id="gpt-4o-mini")
response = await complete(model, context)
```

## Google Gemini

Gemini Developer API:

```python
from agentkit.llm import Context, Model, complete

model = Model(provider="google", api="google-generative-ai", id="gemini-2.5-flash")
context = Context()
context.add_user("Explain embeddings in one sentence.")

response = await complete(model, context)
print(response.text())
```

Vertex AI:

```python
model = Model(
    provider="google-vertex",
    api="google-vertex",
    id="gemini-2.5-flash",
    config={"project": "my-gcp-project", "location": "us-central1"},
)
```

## OpenAI-compatible providers

OpenAI-compatible providers reuse the OpenAI implementation with preset `base_url` values:

```python
from agentkit.llm import openai_compatible_model

model = openai_compatible_model("groq", "llama-3.3-70b-versatile")
model = openai_compatible_model("deepseek", "deepseek-chat")
model = openai_compatible_model("openrouter", "openai/gpt-4o-mini")
model = openai_compatible_model("ollama", "llama3.2")
```

Built-in presets include DeepSeek, Groq, OpenRouter, xAI, Fireworks, Together, Ollama,
Perplexity, Cerebras, SambaNova, and Nebius. You can also pass a custom `base_url`:

```python
model = openai_compatible_model(
    "custom-provider",
    "custom-model",
    base_url="https://example.com/v1",
)
```

## Model registry

AgentKit includes a small model registry and lets you register your own models:

```python
from agentkit.llm import Model, get_model, get_models, register_model

model = get_model("openai", "gpt-4o-mini")

register_model(Model(provider="custom", api="custom-api", id="custom-model"))
custom_models = get_models("custom")
```

The registry is intentionally lightweight, not a generated provider database.

## Runtime options

```python
from agentkit.llm import RunOptions, complete

response = await complete(
    model,
    context,
    RunOptions(
        max_tokens=100,
        temperature=0.2,
        reasoning="high",
        reasoning_budget=8000,
        timeout_ms=30000,
        max_retries=2,
        headers={"x-app": "agentkit"},
    ),
)
```

`timeout_ms` is expressed in milliseconds in AgentKit and converted to provider SDK
seconds internally. Model-level defaults such as `max_tokens` and `headers` can be set
on `Model`; request-level `RunOptions` override them.

### Request hooks and aborts

```python
import asyncio

from agentkit.llm import RunOptions

abort_signal = asyncio.Event()

options = RunOptions(
    abort_signal=abort_signal,
    on_payload=lambda payload, model: {**payload, "temperature": 0.1},
    on_response=lambda response, model: print("provider response received"),
)

# Later, from another task:
abort_signal.set()
```

## Streaming

```python
from agentkit.llm import Context, EventType, Model, stream

model = Model(provider="anthropic", api="anthropic-messages", id="claude-sonnet-4-20250514")
context = Context()
context.add_user("Tell me a short story.")

s = stream(model, context)

async for event in s:
    if event.type == EventType.TEXT_DELTA:
        print(event.data, end="", flush=True)
        # Incremental assistant state is available too:
        # event.partial
        # event.content_index

final_response = await s.result()
```

Stream event names follow a simple lifecycle:

```text
start
text_start, text_delta, text_end
thinking_start, thinking_delta, thinking_end
toolcall_start, toolcall_delta, toolcall_end
done, error
```

Each stream event carries:

```python
event.type
event.data
event.partial        # partial assistant message state
event.content_index  # content block index for content events
```

## Tools live on context

```python
from agentkit.llm import Context, Model, complete, tool


@tool()
def get_weather(city: str, unit: str = "celsius") -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is 22 degrees {unit}."


model = Model(provider="anthropic", api="anthropic-messages", id="claude-sonnet-4-20250514")
context = Context(
    system_prompt="You are a helpful assistant.",
    tools=[get_weather],
)
context.add_user("What's the weather in Paris?")

response = await complete(model, context)

for call in response.tool_calls():
    print(call.name, call.arguments)
```

## Reasoning

Reasoning uses provider-independent levels:

```python
RunOptions(reasoning="minimal")
RunOptions(reasoning="low")
RunOptions(reasoning="medium")
RunOptions(reasoning="high")
RunOptions(reasoning="xhigh", reasoning_budget=32000)
```

Providers map these levels to their own reasoning/thinking parameters.

## Messages and content

Messages have explicit concrete types:

```python
from agentkit.llm import AssistantMessage, Message, TextContent, UserMessage

user = Message.user("hello")
assistant = AssistantMessage(content=[TextContent(text="hi")])
```

The type layer uses Pydantic models, including discriminated content types:

```python
TextContent
ImageContent
ThinkingContent
ToolCall
ToolResult
```

Tool results use a dedicated role so an agent loop can append them cleanly:

```python
context.add_message(response.message)
context.add_tool_result(tool_call_id="call_123", tool_name="search", content="Tool output")
```

Stop reasons are intentionally small:

```text
stop, toolUse, length, error, aborted
```

## Project structure

```text
src/agentkit/
├── llm/
│   ├── api.py
│   ├── context.py
│   ├── model.py
│   ├── models/
│   ├── streaming.py
│   ├── tools.py
│   ├── types/       # Pydantic content, message, and response models
│   └── providers/   # provider registry, built-ins, and integrations
├── agent/           # future agent loop namespace
└── exceptions.py
```

## Provider extension boundary

The top-level `agentkit.llm` package exposes the runtime API. Provider implementation classes live under their own provider packages:

```python
from agentkit.llm.providers import Provider, ModelOptions
from agentkit.llm.providers.openai import OpenAIProvider
from agentkit.llm.providers.google import GoogleProvider
from agentkit.llm.providers.openai_ws import OpenAIWebSocketSession
```

Built-in provider wiring lives in `agentkit.llm.providers.builtins`; the generic provider plugin registry lives in `agentkit.llm.providers.registry`. Built-in model entries live in `agentkit.llm.models.builtins`; the model lookup registry lives in `agentkit.llm.models.registry`.

## Development

```bash
uv sync --extra dev
uv run --extra dev pytest
uv run --extra dev --extra openai pytest
uv run --extra dev ruff check .
uv build
```

See `CONTRIBUTING.md` for contributor guidelines and `SECURITY.md` for vulnerability
reporting.

## Design principles

- One package: `agentkit`
- Model/context-first API for agent-loop usability
- Optional provider dependencies via extras
- Context owns messages and tools
- Pydantic for serializable type models
- Provider-specific code isolated under `agentkit.llm.providers`
- No live API calls in tests
- Keep the core small and explicit

## Status

AgentKit is early-stage. The LLM layer exists. The `agentkit.agent` namespace exists but the agent loop is not implemented yet.
