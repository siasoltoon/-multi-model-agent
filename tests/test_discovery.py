import asyncio

import httpx

from agent_platform.discovery import BUILTIN_PROVIDERS, ProviderDiscovery, ProviderSpec
from agent_platform.provider_api_catalog import CONTRACTS_BY_PROVIDER


def test_builtin_provider_discovers_models_from_catalog(monkeypatch):
    spec = ProviderSpec("demo", "https://demo.test/v1/models", "https://demo.test/v1", "DEMO_API_KEY")
    monkeypatch.setenv("DEMO_API_KEY", "secret")

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, headers=None):
            return httpx.Response(200, request=httpx.Request("GET", url), json={"data": [{"id": "demo-coder", "context_length": 65536, "supported_parameters": ["tools", "tool_choice"], "pricing": {"prompt": "0", "completion": "0"}}]})

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: FakeClient())
    discovery = ProviderDiscovery(providers=(spec,))
    monkeypatch.setattr(discovery, "_discover_ollama", lambda: asyncio.sleep(0, result=[]))
    found = asyncio.run(discovery.discover())
    assert len(found) == 1
    assert found[0].provider == "demo"
    assert found[0].model == "demo-coder"
    assert found[0].context_window == 65536
    assert found[0].tool_support is True
    assert found[0].billing_type == "free"
    assert found[0].api_key_env == "DEMO_API_KEY"


def test_openrouter_requires_explicit_free_model(monkeypatch):
    spec = ProviderSpec("openrouter", "https://openrouter.test/api/v1/models", "https://openrouter.test/api/v1", "OPENROUTER_API_KEY")
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")

    class FakeClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): return False
        async def get(self, url, headers=None):
            return httpx.Response(200, request=httpx.Request("GET", url), json={"data": [
                {"id": "provider/paid-looking", "context_length": 65536, "pricing": {"prompt": "0", "completion": "0"}},
                {"id": "provider/coder:free", "context_length": 65536, "pricing": {"prompt": "0", "completion": "0"}},
            ]})

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: FakeClient())
    discovery = ProviderDiscovery(providers=(spec,))
    monkeypatch.setattr(discovery, "_discover_ollama", lambda: asyncio.sleep(0, result=[]))
    found = asyncio.run(discovery.discover())
    assert [item.model for item in found] == ["provider/coder:free"]
    assert found[0].billing_type == "free"
    assert found[0].metadata["billing_type"] == "free"


def test_openrouter_paid_models_are_never_enabled_by_environment_override(monkeypatch):
    spec = ProviderSpec("openrouter", "https://openrouter.test/api/v1/models", "https://openrouter.test/api/v1", "OPENROUTER_API_KEY")
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    monkeypatch.setenv("AGENT_ALLOW_PAID_OPENROUTER", "true")

    class FakeClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): return False
        async def get(self, url, headers=None):
            return httpx.Response(200, request=httpx.Request("GET", url), json={"data": [{"id": "provider/paid-model", "context_length": 65536, "pricing": {"prompt": "0.1", "completion": "0.2"}}]})

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: FakeClient())
    discovery = ProviderDiscovery(providers=(spec,))
    monkeypatch.setattr(discovery, "_discover_ollama", lambda: asyncio.sleep(0, result=[]))
    found = asyncio.run(discovery.discover())
    assert [item.model for item in found] == ["openrouter/free"]
    assert found[0].billing_type == "free"


def test_provider_without_key_is_not_called(monkeypatch):
    spec = ProviderSpec("demo", "https://demo.test/v1/models", "https://demo.test/v1", "MISSING_API_KEY")
    discovery = ProviderDiscovery(providers=(spec,))
    monkeypatch.setattr(discovery, "_discover_ollama", lambda: asyncio.sleep(0, result=[]))

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("provider should not be called without credentials")

    monkeypatch.setattr(discovery, "_discover_openai_compatible", fail_if_called)
    assert asyncio.run(discovery.discover()) == []


def test_builtin_discovery_urls_match_verified_api_contracts():
    by_provider = {item.name: item for item in BUILTIN_PROVIDERS}
    assert by_provider
    for provider_id, contract in CONTRACTS_BY_PROVIDER.items():
        spec = by_provider[provider_id]
        assert spec.base_url == contract.base_url
        assert spec.models_url == contract.base_url.rstrip("/") + contract.models_path
