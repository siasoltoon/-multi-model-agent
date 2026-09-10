from __future__ import annotations

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
    """Adaptive model router with capability filtering, role-aware scoring and failover."""

    def __init__(self, endpoints: list[ModelEndpoint] | None = None):
        self.endpoints = endpoints or []

    def register(self, endpoint: ModelEndpoint) -> None:
        self.endpoints = [x for x in self.endpoints if x.id != endpoint.id]
        self.endpoints.append(endpoint)

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

    def _candidates(self, *, min_context: int, tools: bool) -> list[ModelEndpoint]:
        now = monotonic()
        candidates = [
            e for e in self.endpoints
            if e.health not in {"OFFLINE", "RATE_LIMITED", "QUOTA_EXHAUSTED"}
            and e.cooldown_until <= now
            and e.context_window >= min_context
            and (not tools or e.tool_support)
            and e.quota_remaining > 0
        ]
        if candidates:
            return candidates
        fallback = [
            e for e in self.endpoints
            if e.health not in {"OFFLINE", "QUOTA_EXHAUSTED"}
            and e.context_window >= min_context
            and (not tools or e.tool_support)
            and e.quota_remaining > 0
        ]
        if not fallback:
            raise RuntimeError("no healthy model endpoint available")
        return fallback

    def ranked(
        self,
        *,
        task_fit: float = 1.0,
        min_context: int = 0,
        tools: bool = False,
        task_type: str = "coding",
        role: str | None = None,
    ) -> list[ModelEndpoint]:
        candidates = self._candidates(min_context=min_context, tools=tools)

        def score(e: ModelEndpoint) -> float:
            explicit_task = self._fit_multiplier(e, "task_fit", task_type)
            effective_task = e.task_fit if explicit_task is None else explicit_task
            role_fit = self._fit_multiplier(e, "role_fit", role) if role else None
            effective_role = 1.0 if role_fit is None else role_fit
            fit = max(0.05, min(1.0, effective_task * task_fit))
            reliability = max(0.05, min(1.0, e.reliability))
            quota = max(0.05, min(1.0, e.quota_remaining))
            speed = self._latency_score(e.latency_ms)
            billing = str(e.metadata.get("billing_type", "unknown")).lower() if isinstance(e.metadata, dict) else "unknown"
            free_bonus = 1.18 if billing in {"free", "local"} else 1.0
            return free_bonus * (fit ** 2) * (effective_role ** 2) * (reliability ** 2) * (0.65 + 0.35 * quota) * (0.75 + 0.25 * speed)

        return sorted(candidates, key=score, reverse=True)

    def choose(self, **kwargs) -> ModelEndpoint:
        return self.ranked(**kwargs)[0]

    def mark_failure(self, endpoint_id: str, error: Exception | str, *, rate_limited: bool = False) -> None:
        text = str(error).lower()
        detected_rate_limit = rate_limited or any(
            marker in text for marker in ("429", "rate limit", "rate_limit", "too many requests", "quota")
        )
        for e in self.endpoints:
            if e.id == endpoint_id:
                e.failures += 1
                e.last_error_at = monotonic()
                e.reliability = max(0.05, e.reliability * 0.85)
                if detected_rate_limit:
                    e.health = "RATE_LIMITED"
                    e.cooldown_until = monotonic() + min(300.0, 15.0 * (2 ** min(e.failures - 1, 4)))
                else:
                    e.health = "DEGRADED" if e.failures < 3 else "ERROR"

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
