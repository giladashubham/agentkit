from __future__ import annotations

import pytest

from agentkit.llm import Tool, tool


@pytest.fixture
def search_tool() -> Tool:
    @tool()
    def search(query: str, max_results: int = 5) -> str:
        """Search the web for information."""
        return f"results for {query}"

    return search
