"""Basic AgentKit completion example.

Set the provider API key expected by the SDK before running, for example
`ANTHROPIC_API_KEY` for Anthropic.
"""

from __future__ import annotations

import asyncio

from agentkit.llm import Context, Model, complete


async def main() -> None:
    model = Model(
        provider="anthropic",
        api="anthropic-messages",
        id="claude-sonnet-4-20250514",
    )
    context = Context(system_prompt="You are a helpful assistant.")
    context.add_user("Say hello in one short sentence.")

    response = await complete(model, context)
    print(response.text())


if __name__ == "__main__":
    asyncio.run(main())
