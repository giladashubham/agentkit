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

## OpenAI

```python
from agentkit.llm import Context, Model, complete

model = Model(provider="openai", api="openai-completions", id="gpt-4o-mini")
context = Context()
context.add_user("Say hello in three words.")

response = await complete(model, context)
print(response.text())
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
│   ├── registry.py
│   ├── streaming.py
│   ├── tools.py
│   ├── types/       # Pydantic content, message, and response models
│   └── providers/   # provider integrations
├── agent/           # future agent loop namespace
└── exceptions.py
```

## Development

```bash
uv sync --extra dev
uv run --extra dev pytest
uv run --extra dev --extra openai pytest
uv run --extra dev ruff check .
```

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
