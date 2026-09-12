import asyncio

import httpx

from agent_platform.discovery import BUILTIN_PROVIDERS, ProviderDiscovery
from agent_platform.provider_api_catalog import get_api_contract
from agent_platform.provider_registry import get_provider


def test_tokenharbor_is_verified_openai_compatible_provider():
    provider = get_provider("tokenharbor")
    contract = get_api_contract("tokenharbor")
    specs = {item.name: item for item in BUILTIN_PROVIDERS}

    assert provider.api_key_env == "TOKENHARBOR_API_KEY"
    assert provider.openai_compatible is True
    assert provider.free_status == "verified"
    assert provider.free_quota == "rolling_7_day_value_allowance"
    assert contract is not None
    assert contract.base_url == "https://tokenharbor.ai/v1"
    assert contract.models_path == "/models"
    assert contract.chat_path == "/chat/completions"
    assert specs["tokenharbor"].models_url == "https://tokenharbor.ai/v1/models"


def test_tokenharbor_discovers_only_the_selected_free_model(monkeypatch):
    monkeypatch.setenv("TOKENHARBOR_API_KEY", "secret")

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, headers=None):
            return httpx.Response(
                200,
                request=httpx.Request("GET", url),
                json={
                    "data": [
                        {
                            "id": "deepseek-v4.1-flash:free",
                            "context_length": 1048576,
                            "supported_parameters": ["tools", "tool_choice"],
                            "pricing": {"prompt": "0", "completion": "0"},
                        },
                        {
                            "id": "deepseek-v4-flash",
                            "context_length": 1048576,
                            "pricing": {"prompt": "0.14", "completion": "0.28"},
                        },
                    ]
                },
            )

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: FakeClient())
    discovery = ProviderDiscovery(providers=tuple(item for item in BUILTIN_PROVIDERS if item.name == "tokenharbor"))
    monkeypatch.setattr(discovery, "_discover_ollama", lambda: asyncio.sleep(0, result=[]))

    found = asyncio.run(discovery.discover())
    assert [(item.provider, item.model) for item in found] == [("tokenharbor", "deepseek-v4.1-flash:free")]
    assert found[0].billing_type == "free"
    assert found[0].metadata["free_route_pinned"] is True


def test_tokenharbor_pinned_free_model_falls_back_when_pricing_is_missing(monkeypatch):
    monkeypatch.setenv("TOKENHARBOR_API_KEY", "secret")

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, headers=None):
            return httpx.Response(
                200,
                request=httpx.Request("GET", url),
                json={"data": [{"id": "deepseek-v4.1-flash:free"}, {"id": "paid-model"}]},
            )

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: FakeClient())
    discovery = ProviderDiscovery(providers=tuple(item for item in BUILTIN_PROVIDERS if item.name == "tokenharbor"))
    monkeypatch.setattr(discovery, "_discover_ollama", lambda: asyncio.sleep(0, result=[]))

    found = asyncio.run(discovery.discover())
    assert [(item.provider, item.model) for item in found] == [("tokenharbor", "deepseek-v4.1-flash:free")]
    assert found[0].metadata["catalog"] == "tokenharbor-free-fallback"
    assert found[0].metadata["free_route_pinned"] is True
    assert found[0].billing_type == "free"


def test_tokenharbor_pinned_free_model_survives_catalog_failure(monkeypatch):
    monkeypatch.setenv("TOKENHARBOR_API_KEY", "secret")

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, headers=None):
            return httpx.Response(
                503,
                request=httpx.Request("GET", url),
                json={"error": "temporary unavailable"},
            )

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: FakeClient())
    discovery = ProviderDiscovery(providers=tuple(item for item in BUILTIN_PROVIDERS if item.name == "tokenharbor"))
    monkeypatch.setattr(discovery, "_discover_ollama", lambda: asyncio.sleep(0, result=[]))

    found = asyncio.run(discovery.discover())
    assert [(item.provider, item.model) for item in found] == [("tokenharbor", "deepseek-v4.1-flash:free")]
    assert found[0].metadata["catalog"] == "tokenharbor-free-fallback"
    assert found[0].metadata["free_route_pinned"] is True
    assert found[0].billing_type == "free"


def test_tokenharbor_free_model_can_be_overridden_without_enabling_paid_models(monkeypatch):
    monkeypatch.setenv("TOKENHARBOR_API_KEY", "secret")
    monkeypatch.setenv("TOKENHARBOR_FREE_MODEL", "another-model:free")

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, headers=None):
            return httpx.Response(
                200,
                request=httpx.Request("GET", url),
                json={
                    "data": [
                        {"id": "deepseek-v4.1-flash:free", "pricing": {"prompt": "0", "completion": "0"}},
                        {"id": "another-model:free", "pricing": {"prompt": "0", "completion": "0"}},
                        {"id": "another-model", "pricing": {"prompt": "0.1", "completion": "0.2"}},
                    ]
                },
            )

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: FakeClient())
    discovery = ProviderDiscovery(providers=tuple(item for item in BUILTIN_PROVIDERS if item.name == "tokenharbor"))
    monkeypatch.setattr(discovery, "_discover_ollama", lambda: asyncio.sleep(0, result=[]))

    found = asyncio.run(discovery.discover())
    assert [(item.provider, item.model) for item in found] == [("tokenharbor", "another-model:free")]
