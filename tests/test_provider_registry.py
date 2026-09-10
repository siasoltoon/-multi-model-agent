from agent_platform.provider_registry import PROVIDER_REGISTRY, get_provider


def test_registry_has_core_provider_categories():
    ids = {item.provider_id for item in PROVIDER_REGISTRY}
    assert {"openrouter", "groq", "mistral", "openai", "anthropic", "gemini", "deepseek", "xai", "ollama"}.issubset(ids)
    assert get_provider("openrouter").category == "gateway"
    assert get_provider("ollama").category == "local"


def test_registry_is_comprehensive_and_unique():
    ids = [item.provider_id for item in PROVIDER_REGISTRY]
    assert len(ids) >= 50
    assert len(ids) == len(set(ids))


def test_native_providers_are_cataloged_before_adapter_work():
    for provider_id in ("anthropic", "gemini", "aws_bedrock", "google_vertex", "azure_foundry", "ibm_watsonx"):
        item = get_provider(provider_id)
        assert item.discovery_supported is False
        assert item.adapter != "openai_compatible"


def test_registry_does_not_mark_candidates_as_permanent_free():
    for item in PROVIDER_REGISTRY:
        if item.free_status != "verified":
            assert item.billing_type != "permanent_free"


def test_registry_metadata_is_safe_to_expose():
    for item in PROVIDER_REGISTRY:
        metadata = item.to_metadata()
        assert "api_key" not in metadata
        assert "secret" not in metadata
        assert metadata["provider_id"] == item.provider_id
