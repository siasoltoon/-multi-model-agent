import pytest

from agent_platform.provider_connections import build_provider_adapter, verified_provider_ids


def test_verified_provider_ids_are_available():
    ids = verified_provider_ids()
    assert len(ids) >= 17
    assert "openrouter" in ids
    assert "cerebras" in ids
    assert "deepseek" in ids


def test_unverified_provider_requires_native_or_verified_contract(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with pytest.raises(ValueError, match="not verified"):
        build_provider_adapter("openai", model="test-model")


def test_verified_provider_uses_contract_base_url(monkeypatch):
    monkeypatch.setenv("CEREBRAS_API_KEY", "test-key")
    adapter = build_provider_adapter("cerebras", model="test-model")
    assert adapter.url == "https://api.cerebras.ai/v1/chat/completions"
    assert adapter.api_key == "test-key"
    assert adapter.model == "test-model"
