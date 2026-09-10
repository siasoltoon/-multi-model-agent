from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

from .adapters import ModelResponse


class NativeProviderError(RuntimeError):
    def __init__(self, provider: str, status_code: int, message: str):
        self.provider = provider
        self.status_code = status_code
        super().__init__(f"{provider}: HTTP {status_code}: {message}")


def _error(response: httpx.Response, provider: str) -> None:
    if response.is_success:
        return
    message = "provider request failed"
    try:
        payload = response.json()
        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, dict):
                message = str(err.get("message") or err.get("type") or message)
            elif err:
                message = str(err)
            elif payload.get("message"):
                message = str(payload["message"])
    except (ValueError, TypeError):
        if response.text:
            message = response.text[:500]
    raise NativeProviderError(provider, response.status_code, message)


def _tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls = []
    for call in message.get("tool_calls", []) or []:
        fn = call.get("function", {})
        calls.append({"id": call.get("id"), "name": fn.get("name", ""), "arguments": fn.get("arguments", "{}")})
    return calls


class OpenAINativeAdapter:
    def __init__(self, api_key: str, model: str, timeout: float = 180.0, base_url: str = "https://api.openai.com/v1"):
        self.api_key, self.model, self.timeout = api_key, model, timeout
        self.url = base_url.rstrip("/") + "/chat/completions"

    async def generate(self, messages, *, tools=None) -> ModelResponse:
        payload = {"model": self.model, "messages": messages}
        if tools:
            payload["tools"] = tools
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.url, headers=headers, json=payload)
            _error(response, "openai")
            data = response.json()
        message = (data.get("choices") or [{}])[0].get("message") or {}
        return ModelResponse(message.get("content") or "", data, data.get("usage", {}), _tool_calls(message))


class AnthropicAdapter:
    def __init__(self, api_key: str, model: str, timeout: float = 180.0, base_url: str = "https://api.anthropic.com"):
        self.api_key, self.model, self.timeout = api_key, model, timeout
        self.url = base_url.rstrip("/") + "/v1/messages"

    async def generate(self, messages, *, tools=None) -> ModelResponse:
        system = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "system")
        converted = [{"role": m["role"], "content": m.get("content", "")} for m in messages if m.get("role") != "system"]
        payload = {"model": self.model, "max_tokens": 4096, "messages": converted}
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [{"name": t.get("function", {}).get("name", ""), "description": t.get("function", {}).get("description", ""), "input_schema": t.get("function", {}).get("parameters", {})} for t in tools]
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.url, headers=headers, json=payload)
            _error(response, "anthropic")
            data = response.json()
        text = ""; calls = []
        for block in data.get("content", []) or []:
            if block.get("type") == "text":
                text += block.get("text", "")
            elif block.get("type") == "tool_use":
                calls.append({"id": block.get("id"), "name": block.get("name", ""), "arguments": json.dumps(block.get("input", {}))})
        usage = data.get("usage", {})
        return ModelResponse(text, data, usage, calls)


class GeminiAdapter:
    def __init__(self, api_key: str, model: str, timeout: float = 180.0, base_url: str = "https://generativelanguage.googleapis.com/v1beta"):
        self.api_key, self.model, self.timeout = api_key, model, timeout
        self.url = base_url.rstrip("/") + f"/models/{model}:generateContent"

    async def generate(self, messages, *, tools=None) -> ModelResponse:
        contents = []
        system_parts = []
        for m in messages:
            role = m.get("role")
            text = str(m.get("content", ""))
            if role == "system":
                system_parts.append(text)
            else:
                contents.append({"role": "model" if role == "assistant" else "user", "parts": [{"text": text}]})
        payload = {"contents": contents}
        if system_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n".join(system_parts)}]}
        if tools:
            declarations = []
            for t in tools:
                fn = t.get("function", {})
                declarations.append({"name": fn.get("name", ""), "description": fn.get("description", ""), "parameters": fn.get("parameters", {})})
            payload["tools"] = [{"functionDeclarations": declarations}]
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.url, params={"key": self.api_key}, json=payload)
            _error(response, "gemini")
            data = response.json()
        parts = (((data.get("candidates") or [{}])[0]).get("content") or {}).get("parts", [])
        text = ""; calls = []
        for part in parts:
            if part.get("text"):
                text += part["text"]
            if part.get("functionCall"):
                fc = part["functionCall"]
                calls.append({"id": fc.get("name"), "name": fc.get("name", ""), "arguments": json.dumps(fc.get("args", {}))})
        return ModelResponse(text, data, data.get("usageMetadata", {}), calls)


def build_native_adapter(provider: str, *, api_key: str, model: str, timeout: float = 180.0):
    if provider == "openai":
        return OpenAINativeAdapter(api_key, model, timeout)
    if provider == "anthropic":
        return AnthropicAdapter(api_key, model, timeout)
    if provider == "gemini":
        return GeminiAdapter(api_key, model, timeout)
    raise ValueError(f"native adapter is not implemented: {provider}")
