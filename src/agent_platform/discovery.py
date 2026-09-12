from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import httpx

from .provider_api_catalog import VERIFIED_PROVIDER_APIS, is_api_verified
from .provider_registry import PROVIDER_REGISTRY


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    models_url: str
    base_url: str
    api_key_env: str
    auth_scheme: str = "bearer"


_registry_by_id = {item.provider_id: item for item in PROVIDER_REGISTRY}
BUILTIN_PROVIDERS: tuple[ProviderSpec, ...] = tuple(
    ProviderSpec(contract.provider_id, contract.base_url.rstrip("/") + contract.models_path, contract.base_url, (_registry_by_id[contract.provider_id].api_key_env or ""), contract.auth_scheme)
    for contract in VERIFIED_PROVIDER_APIS
    if contract.verified and contract.provider_id in _registry_by_id and _registry_by_id[contract.provider_id].api_key_env and _registry_by_id[contract.provider_id].openai_compatible and is_api_verified(contract.provider_id)
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
    """Discover providers while enforcing a hard zero-cost policy."""

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
        return self._finalize(found)

    async def _discover_openai_compatible(self, spec: ProviderSpec, api_key: str) -> list[DiscoveredEndpoint]:
        headers = {"Authorization": f"Bearer {api_key}"}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(spec.models_url, headers=headers)
            response.raise_for_status()
            data = response.json()
        items = data.get("data", []) if isinstance(data, dict) else data
        registry = next((x for x in PROVIDER_REGISTRY if x.provider_id == spec.name), None)
        result: list[DiscoveredEndpoint] = []
        explicit_free_openrouter = False
        tokenharbor_target = os.getenv("TOKENHARBOR_FREE_MODEL", "deepseek-v4.1-flash:free").strip()
        for item in items:
            if not isinstance(item, dict) or not item.get("id"):
                continue
            model = str(item["id"])
            pricing = item.get("pricing")
            is_free = _is_free_pricing(pricing)
            openrouter_free = spec.name == "openrouter" and _is_free_openrouter_model(model)
            if spec.name == "tokenharbor":
                # Token Harbor is intentionally pinned to the user's selected
                # free route. Never discover or route its paid models.
                if model != tokenharbor_target or not is_free:
                    continue
            elif spec.name == "openrouter":
                if not openrouter_free:
                    continue
            elif not is_free:
                continue
            if openrouter_free:
                explicit_free_openrouter = True
            context = int(item.get("context_length") or item.get("context_window") or (registry.default_context_window if registry else 32768))
            supported = item.get("supported_parameters") or []
            tool_support = "tools" in supported or "tool_choice" in supported or not supported
            metadata = {
                "catalog": "openai-compatible",
                "owned_by": item.get("owned_by"),
                "created": item.get("created"),
                "supported_parameters": supported,
                "pricing": pricing,
                "registry": registry.to_metadata() if registry else {},
                "api_verified": True,
                "billing_type": "free",
                "free_status": "verified",
                "zero_cost_verified": True,
            }
            if spec.name == "tokenharbor":
                metadata["free_route"] = tokenharbor_target
                metadata["free_route_pinned"] = True
            result.append(DiscoveredEndpoint(provider=spec.name, model=model, base_url=spec.base_url, context_window=context, tool_support=tool_support, billing_type="free", api_key_env=spec.api_key_env, source=spec.models_url, metadata=metadata))
        if spec.name == "openrouter" and not explicit_free_openrouter:
            result.append(self._openrouter_free_fallback(spec))
        if spec.name == "tokenharbor" and not any(item.provider == "tokenharbor" and item.model == tokenharbor_target for item in result):
            # Some Token Harbor /models responses omit pricing metadata even
            # though the explicitly pinned :free route is valid. Because this
            # exact model is configured as the only allowed Token Harbor route,
            # keep it discoverable without ever admitting arbitrary paid models.
            result.append(self._tokenharbor_free_fallback(spec, tokenharbor_target))
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
            provider = str(item.get("provider", "unknown"))
            billing_type = str(item.get("billing_type", "unknown")).strip().lower()
            metadata = item.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            for model in models:
                if not model:
                    continue
                model_name = str(model)
                if provider == "openrouter":
                    if not _is_free_openrouter_model(model_name):
                        continue
                    catalog_billing = "free"
                else:
                    if not _catalog_item_is_zero_cost(billing_type, metadata):
                        continue
                    catalog_billing = "local" if billing_type == "local" else "free"
                catalog_metadata = dict(metadata)
                catalog_metadata.update({"billing_type": catalog_billing, "zero_cost_verified": True, "free_status": "verified"})
                result.append(DiscoveredEndpoint(provider=provider, model=model_name, base_url=str(base), context_window=int(item.get("context_window", 32768)), tool_support=bool(item.get("tool_support", True)), task_fit=float(item.get("task_fit", 0.8)), reliability=float(item.get("reliability", 0.8)), latency_ms=float(item.get("latency_ms", 1000)), billing_type=catalog_billing, api_key_env=item.get("api_key_env"), source=url, metadata=catalog_metadata))
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
        return [DiscoveredEndpoint(provider="ollama", model=item["name"], base_url=base, context_window=32768, tool_support=True, task_fit=0.75, reliability=0.9, latency_ms=500, billing_type="local", source="ollama:/api/tags", metadata={**item, "billing_type": "local", "zero_cost_verified": True, "category": "local", "free_status": "verified"}) for item in data.get("models", []) if item.get("name")]

    @staticmethod
    def _openrouter_free_fallback(spec: ProviderSpec) -> DiscoveredEndpoint:
        return DiscoveredEndpoint(provider="openrouter", model="openrouter/free", base_url=spec.base_url, context_window=32768, tool_support=True, task_fit=0.8, reliability=0.8, latency_ms=1000.0, billing_type="free", api_key_env=spec.api_key_env, source="openrouter:free-fallback", metadata={"catalog": "openrouter-free-fallback", "billing_type": "free", "free_status": "verified", "zero_cost_verified": True, "api_verified": True})

    @staticmethod
    def _tokenharbor_free_fallback(spec: ProviderSpec, model: str) -> DiscoveredEndpoint:
        return DiscoveredEndpoint(provider="tokenharbor", model=model, base_url=spec.base_url, context_window=32768, tool_support=True, task_fit=0.9, reliability=0.8, latency_ms=800.0, billing_type="free", api_key_env=spec.api_key_env, source="tokenharbor:free-fallback", metadata={"catalog": "tokenharbor-free-fallback", "free_route": model, "free_route_pinned": True, "billing_type": "free", "free_status": "verified", "zero_cost_verified": True, "api_verified": True})

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

    @classmethod
    def _finalize(cls, items: list[DiscoveredEndpoint]) -> list[DiscoveredEndpoint]:
        """Final hard gate: discovery may return only verified zero-cost endpoints."""
        filtered: list[DiscoveredEndpoint] = []
        for item in items:
            metadata = item.metadata if isinstance(item.metadata, dict) else {}
            billing = str(item.billing_type or metadata.get("billing_type", "unknown")).lower()
            is_local = billing == "local" or str(metadata.get("category", "")).lower() == "local"
            is_free = metadata.get("zero_cost_verified") is True or (billing in {"free", "permanent_free"} and _is_free_pricing(metadata.get("pricing")))
            if is_local or is_free:
                filtered.append(item)
        if os.getenv("OPENROUTER_API_KEY", "").strip() and not any(item.provider == "openrouter" for item in filtered):
            spec = next((item for item in BUILTIN_PROVIDERS if item.name == "openrouter"), None)
            if spec:
                filtered.append(cls._openrouter_free_fallback(spec))
        return cls._dedupe(filtered)


def _catalog_item_is_zero_cost(billing_type: str, metadata: dict[str, Any]) -> bool:
    if billing_type == "local":
        return True
    if billing_type not in {"free", "permanent_free"}:
        return False
    return metadata.get("zero_cost_verified") is True or metadata.get("free_status") == "verified" or _is_free_pricing(metadata.get("pricing"))


def _is_free_openrouter_model(model: str) -> bool:
    return model == "openrouter/free" or model.endswith(":free")


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
