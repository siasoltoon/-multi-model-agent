from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx


@dataclass(frozen=True)
class ProviderAPIResult:
    provider: str
    models: list[dict[str, Any]]
    latency_ms: float


@dataclass(frozen=True)
class ChatCompletionResult:
    provider: str
    model: str
    content: str
    raw: dict[str, Any]
    latency_ms: float


class ProviderAPIError(RuntimeError):
    """Normalized provider API error without exposing credentials."""

    def __init__(self, provider: str, status_code: int, message: str, retry_after: float | None = None):
        self.provider = provider
        self.status_code = status_code
        self.retry_after = retry_after
        super().__init__(f"{provider}: HTTP {status_code}: {message}")


class OpenAICompatibleProvider:
    """Shared API client for verified OpenAI-compatible provider contracts."""

    def __init__(self, provider: str, base_url: str, api_key: str, timeout: float = 30.0):
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        message = "provider request failed"
        try:
            payload = response.json()
            if isinstance(payload, dict):
                error = payload.get("error")
                if isinstance(error, dict):
                    message = str(error.get("message") or error.get("code") or message)
                elif error:
                    message = str(error)
                elif payload.get("message"):
                    message = str(payload["message"])
        except (ValueError, TypeError):
            text = response.text.strip()
            if text:
                message = text[:500]
        retry_after = None
        raw_retry = response.headers.get("retry-after")
        if raw_retry:
            try:
                retry_after = max(0.0, float(raw_retry))
            except ValueError:
                retry_after = None
        raise ProviderAPIError(self.provider, response.status_code, message, retry_after)

    async def list_models(self) -> ProviderAPIResult:
        import time

        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            self._raise_for_status(response)
            payload = response.json()
        models = payload.get("data", []) if isinstance(payload, dict) else []
        return ProviderAPIResult(
            provider=self.provider,
            models=[x for x in models if isinstance(x, dict) and x.get("id")],
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    async def health_check(self) -> float:
        import time

        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            self._raise_for_status(response)
        return (time.perf_counter() - started) * 1000

    def _chat_payload(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if extra:
            payload.update(extra)
        return payload

    async def chat_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ChatCompletionResult:
        import time

        payload = self._chat_payload(model, messages, temperature=temperature, max_tokens=max_tokens, tools=tools, tool_choice=tool_choice, extra=extra)
        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.base_url}/chat/completions", headers=self._headers(), json=payload)
            self._raise_for_status(response)
            data = response.json()

        choices = data.get("choices", []) if isinstance(data, dict) else []
        if not choices or not isinstance(choices[0], dict):
            raise ValueError(f"{self.provider}: response contains no choices")
        message = choices[0].get("message") or {}
        content = message.get("content", "") if isinstance(message, dict) else ""
        return ChatCompletionResult(
            provider=self.provider,
            model=model,
            content=content or "",
            raw=data,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    async def stream_chat_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        extra: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        payload = self._chat_payload(model, messages, temperature=temperature, max_tokens=max_tokens, tools=tools, tool_choice=tool_choice, extra=extra)
        payload["stream"] = True
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", f"{self.base_url}/chat/completions", headers=self._headers(), json=payload) as response:
                self._raise_for_status(response)
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = httpx.Response(200, json={}).json() if False else __import__("json").loads(data)
                    except (ValueError, TypeError):
                        continue
                    choices = chunk.get("choices", []) if isinstance(chunk, dict) else []
                    if not choices or not isinstance(choices[0], dict):
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content") if isinstance(delta, dict) else None
                    if content:
                        yield str(content)


def build_openai_compatible(provider: str, base_url: str, api_key: str, timeout: float = 30.0) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(provider, base_url, api_key, timeout)
