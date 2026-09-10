import asyncio

import httpx

from agent_platform.discovery import ProviderDiscovery, ProviderSpec


def test_builtin_provider_discovers_models_from_catalog(monkeypatch):
    spec = ProviderSpec("demo", "https://demo.test/v1/models", "https://demo.test/v1", "DEMO_API_KEY")
    monkeypatch.setenv("DEMO_API_KEY", "secret")

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, headers=None):
            assert url == spec.models_url
            assert headers == {"Authorization": "Bearer secret"}
            return httpx.Response(
                200,
                request=httpx.Request("GET", url),
                json={
                    "data": [
                        {
                            "id": "demo-coder",
                            "context_length": 65536,
                            "supported_parameters": ["tools", "tool_choice"],
                            "pricing": {"prompt": "0", "completion": "0"},
                        }
                    ]
                },
            )

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


def test_provider_without_key_is_not_called(monkeypatch):
    spec = ProviderSpec("demo", "https://demo.test/v1/models", "https://demo.test/v1", "MISSING_API_KEY")
    discovery = ProviderDiscovery(providers=(spec,))
    monkeypatch.setattr(discovery, "_discover_ollama", lambda: asyncio.sleep(0, result=[]))

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("provider should not be called without credentials")

    monkeypatch.setattr(discovery, "_discover_openai_compatible", fail_if_called)
    assert asyncio.run(discovery.discover()) == []
