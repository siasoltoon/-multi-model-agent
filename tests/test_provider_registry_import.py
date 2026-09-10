def test_provider_registry_imports_and_exposes_core_entries():
    from agent_platform.provider_registry import PROVIDER_REGISTRY, get_provider

    assert len(PROVIDER_REGISTRY) >= 30
    assert get_provider("openrouter").category == "gateway"
    assert get_provider("ollama").billing_type == "local"
