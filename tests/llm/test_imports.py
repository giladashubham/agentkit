from __future__ import annotations

import sys


def test_llm_import_does_not_import_provider_sdks() -> None:
    sys.modules.pop("anthropic", None)
    sys.modules.pop("openai", None)

    import agentkit.llm as llm

    assert llm.Model.__name__ == "Model"
    assert llm.Context.__name__ == "Context"
    assert llm.complete.__name__ == "complete"
    assert llm.stream.__name__ == "stream"
    assert "anthropic" not in sys.modules
    assert "openai" not in sys.modules


def test_client_api_is_not_exported() -> None:
    import agentkit.llm as llm

    assert not hasattr(llm, "LLM")
    assert not hasattr(llm, "openai")
    assert not hasattr(llm, "anthropic")
