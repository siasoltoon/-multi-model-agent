import asyncio

import httpx

from agent_platform.provider_adapters import OpenAICompatibleProvider


def test_openai_compatible_lists_models_and_measures_latency(monkeypatch):
    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, headers=None):
            assert url == "https://demo.test/v1/models"
            assert headers == {"Authorization": "Bearer secret"}
            return httpx.Response(
                200,
                request=httpx.Request("GET", url),
                json={"data": [{"id": "demo-coder"}, {"id": "demo-chat"}]},
            )

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: FakeClient())
    provider = OpenAICompatibleProvider("demo", "https://demo.test/v1", "secret")
    result = asyncio.run(provider.list_models())

    assert result.provider == "demo"
    assert [m["id"] for m in result.models] == ["demo-coder", "demo-chat"]
    assert result.latency_ms >= 0
