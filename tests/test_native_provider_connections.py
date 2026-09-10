from agent_platform.native_adapters import AnthropicAdapter, GeminiAdapter, OpenAINativeAdapter
from agent_platform.provider_connections import build_provider_adapter


def test_openai_connection_uses_native_adapter(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    adapter = build_provider_adapter("openai", model="gpt-test")
    assert isinstance(adapter, OpenAINativeAdapter)
    assert adapter.model == "gpt-test"


def test_anthropic_connection_uses_native_adapter(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    adapter = build_provider_adapter("anthropic", model="claude-test")
    assert isinstance(adapter, AnthropicAdapter)
    assert adapter.model == "claude-test"


def test_gemini_connection_uses_native_adapter(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    adapter = build_provider_adapter("gemini", model="gemini-test")
    assert isinstance(adapter, GeminiAdapter)
    assert adapter.model == "gemini-test"
