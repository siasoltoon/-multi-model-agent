from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import httpx

from .provider_api_catalog import is_api_verified
from .provider_registry import PROVIDER_REGISTRY


@dataclass(frozen=True)
class ProviderSpec:
    """Backwards-compatible provider discovery specification."""

    name: str
    models_url: str
    base_url: str
    api_key_env: str
    auth_scheme: str = "bearer"


# Registry entries are catalog metadata. Live discovery is restricted to API
# contracts explicitly verified in provider_api_catalog.py. A provider no
# longer needs a second discovery flag: the verified contract is the gate.
BUILTIN_PROVIDERS: tuple[ProviderSpec, ...] = tuple(
    ProviderSpec(
        item.provider_id,
        item.models_url or "",
        item.base_url or "",
        item.api_key_env or "",
        item.auth_scheme,
    )
    for item in PROVIDER_REGISTRY
    if (
        item.models_url
        and item.base_url
        and item.api_key_env
        and item.openai_compatible
        and is_api_verified(item.provider_id)
    )
)


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
    """Discover configured providers from the central registry plus local Ollama."""

    def __init__(self, catalog_urls: list[str] | None = None, timeout: float = 15.0, providers: tuple[ProviderSpec, ...] = BUILTIN_PROVIDERS):
        self.catalog_urls = catalog_urls if catalog_urls is not None else self._env_catalogs()
        self.providers = providers
        self.timeout = timeout

    @staticmethod
    def _env_catalogs() -> list[str]:
        raw = os.getenv("AGENT_PROVIDER_CATALOGS", "")
        return [x.strip() for x in raw.split(",") if x.strip()]

    async def discover(self) -> list[DiscoveredEndpoint]:
        found: list[DiscoveredEndpoint] = []
        for spec in self.providers:
            api_key = os.getenv(spec.api_key_env, "").strip()
            if not api_key:
                continue
            try:
                found.extend(await self._discover_openai_compatible(spec, api_key))
            except Exception:
                continue
        for url in self.catalog_urls:
            try:
                found.extend(await self._load_catalog(url))
            except Exception:
                continue
        found.extend(await self._discover_ollama())
        return self._dedupe(found)

    async def _discover_openai_compatible(self, spec: ProviderSpec, api_key: str) -> list[DiscoveredEndpoint]:
        headers = {"Authorization": f"Bearer {api_key}"}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(spec.models_url, headers=headers)
            response.raise_for_status()
            data = response.json()
        items = data.get("data", []) if isinstance(data, dict) else data
        registry = next((x for x in PROVIDER_REGISTRY if x.provider_id == spec.name), None)
        result: list[DiscoveredEndpoint] = []
        for item in items:
            if not isinstance(item, dict) or not item.get("id"):
                continue
            model = str(item["id"])
            context = int(item.get("context_length") or item.get("context_window") or (registry.default_context_window if registry else 32768))
            supported = item.get("supported_parameters") or []
            tool_support = "tools" in supported or "tool_choice" in supported or not supported
            metadata = {
                "catalog": "openai-compatible",
                "owned_by": item.get("owned_by"),
                "created": item.get("created"),
                "supported_parameters": supported,
                "pricing": item.get("pricing"),
                "registry": registry.to_metadata() if registry else {},
                "api_verified": True,
            }
            result.append(DiscoveredEndpoint(
                provider=spec.name,
                model=model,
                base_url=spec.base_url,
                context_window=context,
                tool_support=tool_support,
                billing_type=(registry.billing_type if registry else ("free" if _is_free_pricing(item.get("pricing")) else "paid_or_unknown")),
                api_key_env=spec.api_key_env,
                source=spec.models_url,
                metadata=metadata,
            ))
        return result

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
                    provider=str(item.get("provider", "unknown")), model=str(model), base_url=str(base),
                    context_window=int(item.get("context_window", 32768)), tool_support=bool(item.get("tool_support", True)),
                    task_fit=float(item.get("task_fit", 0.8)), reliability=float(item.get("reliability", 0.8)),
                    latency_ms=float(item.get("latency_ms", 1000)), billing_type=str(item.get("billing_type", "unknown")),
                    api_key_env=item.get("api_key_env"), source=url, metadata=item.get("metadata", {}),
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
            provider="ollama", model=item["name"], base_url=base, context_window=32768,
            tool_support=True, task_fit=0.75, reliability=0.9, latency_ms=500, billing_type="local",
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


def _is_free_pricing(pricing: Any) -> bool:
    if pricing is None:
        return False
    if isinstance(pricing, (int, float)):
        return float(pricing) == 0
    if isinstance(pricing, dict):
        values = [pricing.get(k) for k in ("prompt", "completion", "input", "output")]
        present = [v for v in values if v is not None]
        if not present:
            return False
        try:
            return all(float(v) == 0 for v in present)
        except (TypeError, ValueError):
            return False
    return False


def env_api_key(env_name: str | None) -> str | None:
    return os.getenv(env_name) if env_name else None
