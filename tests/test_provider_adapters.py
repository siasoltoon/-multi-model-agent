import asyncio

import httpx

from agent_platform.provider_adapters import OpenAICompatibleProvider, ProviderAPIError


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


def test_openai_compatible_chat_completion(monkeypatch):
    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, headers=None, json=None):
            assert url == "https://demo.test/v1/chat/completions"
            assert headers["Authorization"] == "Bearer secret"
            assert json["model"] == "demo-coder"
            assert json["tools"][0]["type"] == "function"
            return httpx.Response(
                200,
                request=httpx.Request("POST", url),
                json={"choices": [{"message": {"role": "assistant", "content": "done"}}]},
            )

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: FakeClient())
    provider = OpenAICompatibleProvider("demo", "https://demo.test/v1", "secret")
    result = asyncio.run(
        provider.chat_completion(
            "demo-coder",
            [{"role": "user", "content": "test"}],
            tools=[{"type": "function", "function": {"name": "run"}}],
        )
    )

    assert result.provider == "demo"
    assert result.model == "demo-coder"
    assert result.content == "done"
    assert result.raw["choices"][0]["message"]["content"] == "done"
    assert result.latency_ms >= 0


def test_provider_api_error_normalizes_rate_limit_and_retry_after(monkeypatch):
    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, headers=None):
            return httpx.Response(
                429,
                request=httpx.Request("GET", url),
                headers={"Retry-After": "7"},
                json={"error": {"message": "too many requests"}},
            )

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: FakeClient())
    provider = OpenAICompatibleProvider("demo", "https://demo.test/v1", "secret")

    try:
        asyncio.run(provider.list_models())
        raise AssertionError("expected ProviderAPIError")
    except ProviderAPIError as exc:
        assert exc.provider == "demo"
        assert exc.status_code == 429
        assert exc.retry_after == 7.0
        assert "secret" not in str(exc)


def test_openai_compatible_streaming_extracts_sse_content(monkeypatch):
    class FakeResponse:
        def __init__(self):
            self.is_success = True

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def aiter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"hel"}}]}'
            yield 'data: {"choices":[{"delta":{"content":"lo"}}]}'
            yield "data: [DONE]"

    class FakeStreamClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def stream(self, method, url, headers=None, json=None):
            assert method == "POST"
            assert url == "https://demo.test/v1/chat/completions"
            assert json["stream"] is True
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: FakeStreamClient())
    provider = OpenAICompatibleProvider("demo", "https://demo.test/v1", "secret")

    async def collect():
        return [chunk async for chunk in provider.stream_chat_completion("demo-coder", [{"role": "user", "content": "hi"}])]

    assert asyncio.run(collect()) == ["hel", "lo"]
