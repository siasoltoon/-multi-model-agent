import asyncio

import httpx

from agent_platform.discovery import ProviderDiscovery, ProviderSpec


def test_openrouter_falls_back_to_free_router_when_catalog_has_no_explicit_free_model(monkeypatch):
    spec = ProviderSpec("openrouter", "https://openrouter.test/api/v1/models", "https://openrouter.test/api/v1", "OPENROUTER_API_KEY")
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    monkeypatch.delenv("AGENT_ALLOW_PAID_OPENROUTER", raising=False)

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, headers=None):
            return httpx.Response(
                200,
                request=httpx.Request("GET", url),
                json={"data": [{"id": "~openai/gpt-astra-latest", "context_length": 65536}]},
            )

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: FakeClient())
    discovery = ProviderDiscovery(providers=(spec,))
    monkeypatch.setattr(discovery, "_discover_ollama", lambda: asyncio.sleep(0, result=[]))

    found = asyncio.run(discovery.discover())

    assert [item.model for item in found] == ["openrouter/free"]
    assert found[0].provider == "openrouter"
    assert found[0].billing_type == "free"
    assert found[0].metadata["billing_type"] == "free"
