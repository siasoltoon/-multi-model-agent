import asyncio

import httpx

from agent_platform.health import ProviderHealthMonitor
from agent_platform.router import ModelEndpoint, SmartRouter


def test_probe_marks_online_and_records_latency(monkeypatch):
    router = SmartRouter([ModelEndpoint("m", "openai", "x", "https://example.test")])
    monitor = ProviderHealthMonitor(router)

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, headers=None):
            return httpx.Response(200, request=httpx.Request("GET", url), headers={"x-ratelimit-remaining": "80", "x-ratelimit-limit": "100"})

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: FakeClient())
    result = asyncio.run(monitor.probe(router.endpoints[0]))

    assert result.health == "ONLINE"
    assert result.status_code == 200
    assert router.endpoints[0].quota_remaining == 0.8
    assert router.endpoints[0].latency_ms >= 0


def test_probe_marks_rate_limited(monkeypatch):
    endpoint = ModelEndpoint("m", "openai", "x", "https://example.test")
    router = SmartRouter([endpoint])
    monitor = ProviderHealthMonitor(router)

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, headers=None):
            return httpx.Response(429, request=httpx.Request("GET", url), headers={"retry-after": "30"})

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: FakeClient())
    result = asyncio.run(monitor.probe(endpoint))

    assert result.health == "RATE_LIMITED"
    assert endpoint.health == "RATE_LIMITED"
    assert endpoint.cooldown_until > 0


def test_ollama_probe_uses_tags_endpoint(monkeypatch):
    endpoint = ModelEndpoint("m", "ollama", "qwen", "http://127.0.0.1:11434")
    router = SmartRouter([endpoint])
    monitor = ProviderHealthMonitor(router)
    seen = []

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, headers=None):
            seen.append(url)
            return httpx.Response(200, request=httpx.Request("GET", url), json={"models": []})

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: FakeClient())
    asyncio.run(monitor.probe(endpoint))

    assert seen == ["http://127.0.0.1:11434/api/tags"]
