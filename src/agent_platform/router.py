from __future__ import annotations

import os
from dataclasses import dataclass, field
from time import monotonic


@dataclass
class ModelEndpoint:
    id: str
    provider: str
    model: str
    base_url: str = ""
    context_window: int = 32768
    tool_support: bool = True
    task_fit: float = 0.8
    reliability: float = 0.8
    latency_ms: float = 1000.0
    quota_remaining: float = 1.0
    health: str = "ONLINE"
    api_key_env: str | None = None
    metadata: dict = field(default_factory=dict)
    failures: int = 0
    last_error_at: float | None = None
    cooldown_until: float = 0.0


class SmartRouter:
    """Adaptive router with capability filtering, zero-cost enforcement and failover."""

    def __init__(self, endpoints: list[ModelEndpoint] | None = None):
        self.endpoints = endpoints or []

    def register(self, endpoint: ModelEndpoint) -> None:
        self.endpoints = [x for x in self.endpoints if x.id != endpoint.id]
        self.endpoints.append(endpoint)

    @staticmethod
    def is_zero_cost(endpoint: ModelEndpoint) -> bool:
        """Return True only when the endpoint is explicitly proven cost-free.

        This is intentionally a hard safety gate, not a ranking preference.
        Unknown, trial, paid, and merely "candidate free" providers are never
        eligible. Local runtimes are always zero-cost at the inference layer.
        """
        metadata = endpoint.metadata if isinstance(endpoint.metadata, dict) else {}
        billing = str(metadata.get("billing_type", "unknown")).strip().lower()
        category = str(metadata.get("category", "")).strip().lower()
        free_status = str(metadata.get("free_status", "unknown")).strip().lower()
        zero_cost_verified = metadata.get("zero_cost_verified") is True

        if billing == "local" or category == "local":
            return True
        if billing not in {"free", "permanent_free"}:
            return False
        if zero_cost_verified or free_status == "verified":
            return True

        pricing = metadata.get("pricing")
        if isinstance(pricing, dict):
            values = [pricing.get(k) for k in ("prompt", "completion", "input", "output")]
            present = [v for v in values if v is not None]
            if present:
                try:
                    return all(float(v) == 0 for v in present)
                except (TypeError, ValueError):
                    return False
        return False

    @staticmethod
    def _fit_multiplier(endpoint: ModelEndpoint, key: str, value: str) -> float | None:
        capabilities = endpoint.metadata.get(key, {}) if isinstance(endpoint.metadata, dict) else {}
        if isinstance(capabilities, dict) and value in capabilities:
            try:
                return max(0.05, min(1.0, float(capabilities[value])))
            except (TypeError, ValueError):
                pass
        return None

    @staticmethod
    def _latency_score(latency_ms: float) -> float:
        return 1.0 / (1.0 + max(latency_ms, 0.0) / 1000.0)

    @staticmethod
    def _provider_priority(endpoint: ModelEndpoint) -> float:
        metadata = endpoint.metadata if isinstance(endpoint.metadata, dict) else {}
        billing = str(metadata.get("billing_type", "unknown")).lower()
        category = str(metadata.get("category", "")).lower()
        if billing == "local" or category == "local":
            return 1.12
        if billing in {"free", "permanent_free"}:
            return 0.95
        return 0.0

    def _candidates(self, *, min_context: int, tools: bool) -> list[ModelEndpoint]:
        now = monotonic()
        candidates = [
            e for e in self.endpoints
            if self.is_zero_cost(e)
            and e.health not in {"OFFLINE", "RATE_LIMITED", "QUOTA_EXHAUSTED"}
            and e.cooldown_until <= now
            and e.context_window >= min_context
            and (not tools or e.tool_support)
            and e.quota_remaining > 0
        ]
        if candidates:
            return candidates
        fallback = [
            e for e in self.endpoints
            if self.is_zero_cost(e)
            and e.health not in {"OFFLINE", "QUOTA_EXHAUSTED"}
            and e.context_window >= min_context
            and (not tools or e.tool_support)
            and e.quota_remaining > 0
        ]
        if not fallback:
            raise RuntimeError("no healthy zero-cost model endpoint available")
        return fallback

    def _score(self, e: ModelEndpoint, *, task_fit: float, task_type: str, role: str | None) -> float:
        explicit_task = self._fit_multiplier(e, "task_fit", task_type)
        effective_task = e.task_fit if explicit_task is None else explicit_task
        role_fit = self._fit_multiplier(e, "role_fit", role) if role else None
        effective_role = 1.0 if role_fit is None else role_fit
        fit = max(0.05, min(1.0, effective_task * task_fit))
        reliability = max(0.05, min(1.0, e.reliability))
        quota = max(0.05, min(1.0, e.quota_remaining))
        speed = self._latency_score(e.latency_ms)
        return self._provider_priority(e) * (fit ** 2) * (effective_role ** 2) * (reliability ** 2) * (0.65 + 0.35 * quota) * (0.75 + 0.25 * speed)

    def ranked(self, *, task_fit: float = 1.0, min_context: int = 0, tools: bool = False, task_type: str = "coding", role: str | None = None) -> list[ModelEndpoint]:
        candidates = self._candidates(min_context=min_context, tools=tools)
        return sorted(candidates, key=lambda e: self._score(e, task_fit=task_fit, task_type=task_type, role=role), reverse=True)

    def ranked_provider_diverse(self, *, task_fit: float = 1.0, min_context: int = 0, tools: bool = False, task_type: str = "coding", role: str | None = None, max_providers: int | None = None) -> list[ModelEndpoint]:
        """Return the best zero-cost endpoint from each provider for failover."""
        ranked = self.ranked(task_fit=task_fit, min_context=min_context, tools=tools, task_type=task_type, role=role)
        selected: list[ModelEndpoint] = []
        seen: set[str] = set()
        for endpoint in ranked:
            if endpoint.provider in seen:
                continue
            selected.append(endpoint)
            seen.add(endpoint.provider)
            if max_providers is not None and len(selected) >= max(1, int(max_providers)):
                break
        return selected

    def choose(self, **kwargs) -> ModelEndpoint:
        return self.ranked(**kwargs)[0]

    def choose_provider_diverse(self, **kwargs) -> ModelEndpoint:
        return self.ranked_provider_diverse(**kwargs)[0]

    def mark_failure(self, endpoint_id: str, error: Exception | str, *, rate_limited: bool = False, provider_quota_exhausted: bool = False) -> None:
        text = str(error).lower()
        detected_quota = provider_quota_exhausted or any(marker in text for marker in ("free-models-per-day", "daily quota", "daily limit", "quota exhausted", "quota_exceeded", "insufficient_quota", "billing limit"))
        detected_rate_limit = rate_limited or any(marker in text for marker in ("429", "rate limit", "rate_limit", "too many requests", "quota"))
        failed = next((e for e in self.endpoints if e.id == endpoint_id), None)
        if failed is None:
            return
        failed.failures += 1
        failed.last_error_at = monotonic()
        failed.reliability = max(0.05, failed.reliability * 0.85)
        if detected_quota:
            cooldown = max(60.0, float(os.getenv("AGENT_PROVIDER_QUOTA_COOLDOWN_SECONDS", "86400")))
            until = monotonic() + cooldown
            for endpoint in self.endpoints:
                if endpoint.provider == failed.provider:
                    endpoint.health = "QUOTA_EXHAUSTED"
                    endpoint.quota_remaining = 0.0
                    endpoint.cooldown_until = until
            return
        if detected_rate_limit:
            failed.health = "RATE_LIMITED"
            failed.cooldown_until = monotonic() + min(300.0, 15.0 * (2 ** min(failed.failures - 1, 4)))
        else:
            failed.health = "DEGRADED" if failed.failures < 3 else "ERROR"

    def mark_success(self, endpoint_id: str, *, latency_ms: float | None = None) -> None:
        for e in self.endpoints:
            if e.id == endpoint_id:
                e.failures = 0
                e.reliability = min(1.0, e.reliability * 1.03)
                e.health = "ONLINE"
                e.cooldown_until = 0.0
                if latency_ms is not None:
                    e.latency_ms = latency_ms

    def update_health(self, endpoint_id: str, health: str, *, latency_ms: float | None = None, quota_remaining: float | None = None) -> None:
        for e in self.endpoints:
            if e.id == endpoint_id:
                e.health = health
                if latency_ms is not None:
                    e.latency_ms = latency_ms
                if quota_remaining is not None:
                    e.quota_remaining = max(0.0, min(1.0, quota_remaining))
                if health == "ONLINE":
                    e.cooldown_until = 0.0
