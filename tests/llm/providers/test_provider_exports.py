from __future__ import annotations


def test_providers_package_exports_only_base_registry_api() -> None:
    import agentkit.llm.providers as providers

    assert providers.Provider.__name__ == "Provider"
    assert providers.ModelOptions.__name__ == "ModelOptions"
    assert providers.register_provider.__name__ == "register_provider"
    assert not hasattr(providers, "OpenAIProvider")
    assert not hasattr(providers, "AnthropicProvider")
    assert not hasattr(providers, "OpenAIWebSocketSession")


def test_top_level_llm_does_not_export_provider_classes() -> None:
    import agentkit.llm as llm

    assert not hasattr(llm, "OpenAIProvider")
    assert not hasattr(llm, "AnthropicProvider")
    assert not hasattr(llm, "OpenAIWebSocketSession")
