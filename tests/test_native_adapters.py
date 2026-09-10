import json

import httpx
import pytest

from agent_platform.native_adapters import AnthropicAdapter, GeminiAdapter, OpenAINativeAdapter, NativeProviderError


class MockTransport(httpx.AsyncBaseTransport):
    def __init__(self, handler):
        self.handler = handler

    async def handle_async_request(self, request):
        return self.handler(request)


def response(request, payload, status=200):
    return httpx.Response(status, json=payload, request=request)


def patch_client(monkeypatch, transport):
    original = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = transport
        return original(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)


@pytest.mark.asyncio
async def test_openai_native_tool_calls(monkeypatch):
    def handler(request):
        body = json.loads(request.content)
        assert body["model"] == "gpt-test"
        assert body["tools"]
        return response(request, {"choices": [{"message": {"content": "", "tool_calls": [{"id": "1", "function": {"name": "run", "arguments": "{}"}}]}}], "usage": {"total_tokens": 3}})

    patch_client(monkeypatch, MockTransport(handler))
    result = await OpenAINativeAdapter("key", "gpt-test").generate([{"role": "user", "content": "test"}], tools=[{"type": "function", "function": {"name": "run", "parameters": {}}}])
    assert result.tool_calls[0]["name"] == "run"


@pytest.mark.asyncio
async def test_anthropic_native_response(monkeypatch):
    def handler(request):
        body = json.loads(request.content)
        assert body["system"] == "system"
        assert body["messages"][0]["role"] == "user"
        return response(request, {"content": [{"type": "text", "text": "ok"}], "usage": {"input_tokens": 1}})

    patch_client(monkeypatch, MockTransport(handler))
    result = await AnthropicAdapter("key", "claude-test").generate([{"role": "system", "content": "system"}, {"role": "user", "content": "hi"}])
    assert result.text == "ok"


@pytest.mark.asyncio
async def test_gemini_native_function_call(monkeypatch):
    def handler(request):
        assert "key" in str(request.url)
        return response(request, {"candidates": [{"content": {"parts": [{"functionCall": {"name": "run", "args": {"x": 1}}}]}}], "usageMetadata": {"totalTokenCount": 4}})

    patch_client(monkeypatch, MockTransport(handler))
    result = await GeminiAdapter("key", "gemini-test").generate([{"role": "user", "content": "hi"}])
    assert result.tool_calls[0]["name"] == "run"
    assert json.loads(result.tool_calls[0]["arguments"])["x"] == 1


@pytest.mark.asyncio
async def test_native_errors_are_normalized(monkeypatch):
    def handler(request):
        return response(request, {"error": {"message": "bad key"}}, status=401)

    patch_client(monkeypatch, MockTransport(handler))
    with pytest.raises(NativeProviderError, match="anthropic: HTTP 401"):
        await AnthropicAdapter("key", "claude-test").generate([{"role": "user", "content": "hi"}])
