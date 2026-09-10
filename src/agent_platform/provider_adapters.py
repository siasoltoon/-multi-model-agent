from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class ProviderAPIResult:
    provider: str
    models: list[dict[str, Any]]
    latency_ms: float


class OpenAICompatibleProvider:
    """Small shared client for providers exposing the OpenAI-compatible API shape."""

    def __init__(self, provider: str, base_url: str, api_key: str, timeout: float = 30.0):
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

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


def build_openai_compatible(provider: str, base_url: str, api_key: str, timeout: float = 30.0) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(provider, base_url, api_key, timeout)
