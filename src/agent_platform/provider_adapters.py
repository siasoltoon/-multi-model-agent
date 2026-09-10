from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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


class OpenAICompatibleProvider:
    """Shared client for verified providers exposing the OpenAI API shape."""

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

    async def list_models(self) -> ProviderAPIResult:
        import time

        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            response.raise_for_status()
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
            response.raise_for_status()
        return (time.perf_counter() - started) * 1000

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

        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            )
            response.raise_for_status()
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


def build_openai_compatible(provider: str, base_url: str, api_key: str, timeout: float = 30.0) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(provider, base_url, api_key, timeout)
