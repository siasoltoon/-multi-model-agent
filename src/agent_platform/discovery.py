from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class DiscoveredEndpoint:
    provider: str
    model: str
    base_url: str
    context_window: int = 32768
    tool_support: bool = True
    task_fit: float = 0.8
    reliability: float = 0.8
    latency_ms: float = 1000.0
    billing_type: str = "unknown"
    api_key_env: str | None = None
    source: str = "runtime"
    metadata: dict[str, Any] = field(default_factory=dict)


class ProviderDiscovery:
    """Discovers declared provider catalogs and locally available Ollama models.

    It never scrapes credentials or invents API keys. Provider catalogs are ordinary
    JSON endpoints configured by the operator; credentials come only from env vars.
    """

    def __init__(self, catalog_urls: list[str] | None = None, timeout: float = 15.0):
        self.catalog_urls = catalog_urls or self._env_catalogs()
        self.timeout = timeout

    @staticmethod
    def _env_catalogs() -> list[str]:
        raw = os.getenv("AGENT_PROVIDER_CATALOGS", "")
        return [x.strip() for x in raw.split(",") if x.strip()]

    async def discover(self) -> list[DiscoveredEndpoint]:
        found: list[DiscoveredEndpoint] = []
        for url in self.catalog_urls:
            try:
                found.extend(await self._load_catalog(url))
            except Exception:
                continue
        found.extend(await self._discover_ollama())
        return self._dedupe(found)

    async def _load_catalog(self, url: str) -> list[DiscoveredEndpoint]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
        items = data.get("providers", data if isinstance(data, list) else [])
        result: list[DiscoveredEndpoint] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            base = item.get("base_url") or item.get("endpoint")
            models = item.get("models") or [item.get("model")]
            if not base or not models:
                continue
            for model in models:
                if not model:
                    continue
                result.append(DiscoveredEndpoint(
                    provider=str(item.get("provider", "unknown")),
                    model=str(model),
                    base_url=str(base),
                    context_window=int(item.get("context_window", 32768)),
                    tool_support=bool(item.get("tool_support", True)),
                    task_fit=float(item.get("task_fit", 0.8)),
                    reliability=float(item.get("reliability", 0.8)),
                    latency_ms=float(item.get("latency_ms", 1000)),
                    billing_type=str(item.get("billing_type", "unknown")),
                    api_key_env=item.get("api_key_env"),
                    source=url,
                    metadata=item.get("metadata", {}),
                ))
        return result

    async def _discover_ollama(self) -> list[DiscoveredEndpoint]:
        base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                response = await client.get(base + "/api/tags")
                response.raise_for_status()
                data = response.json()
        except Exception:
            return []
        return [DiscoveredEndpoint(
            provider="ollama", model=item["name"], base_url=base,
            context_window=32768, tool_support=True, task_fit=0.75,
            reliability=0.9, latency_ms=500, billing_type="local",
            source="ollama:/api/tags", metadata=item,
        ) for item in data.get("models", []) if item.get("name")]

    @staticmethod
    def _dedupe(items: list[DiscoveredEndpoint]) -> list[DiscoveredEndpoint]:
        seen: set[tuple[str, str, str]] = set()
        result = []
        for item in items:
            key = (item.provider, item.model, item.base_url)
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result


def env_api_key(env_name: str | None) -> str | None:
    """Resolve a provider secret without ever storing the secret in the catalog."""
    return os.getenv(env_name) if env_name else None
