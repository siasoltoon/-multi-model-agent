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
    """Adaptive model router with capability filtering, health cooldowns and scoring."""

    def __init__(self, endpoints: list[ModelEndpoint] | None = None):
        self.endpoints = endpoints or []

    def register(self, endpoint: ModelEndpoint) -> None:
        self.endpoints = [x for x in self.endpoints if x.id != endpoint.id]
        self.endpoints.append(endpoint)

    @staticmethod
    def _task_multiplier(endpoint: ModelEndpoint, task_type: str) -> float:
        capabilities = endpoint.metadata.get("task_fit", {}) if isinstance(endpoint.metadata, dict) else {}
        if isinstance(capabilities, dict) and task_type in capabilities:
            try:
                return max(0.05, min(1.0, float(capabilities[task_type])))
            except (TypeError, ValueError):
                pass
        return 1.0

    @staticmethod
    def _latency_score(latency_ms: float) -> float:
        return 1.0 / (1.0 + max(latency_ms, 0.0) / 1000.0)

    def choose(
        self,
        *,
        task_fit: float = 1.0,
        min_context: int = 0,
        tools: bool = False,
        task_type: str = "coding",
    ) -> ModelEndpoint:
        now = monotonic()
        candidates = [
            e for e in self.endpoints
            if e.health not in {"OFFLINE", "RATE_LIMITED", "QUOTA_EXHAUSTED"}
            and e.cooldown_until <= now
            and e.context_window >= min_context
            and (not tools or e.tool_support)
            and e.quota_remaining > 0
        ]
        if not candidates:
            fallback = [
                e for e in self.endpoints
                if e.health not in {"OFFLINE", "QUOTA_EXHAUSTED"}
                and e.context_window >= min_context
                and (not tools or e.tool_support)
                and e.quota_remaining > 0
            ]
            if fallback:
                candidates = fallback
            else:
                raise RuntimeError("no healthy model endpoint available")

        def score(e: ModelEndpoint) -> float:
            fit = max(0.05, min(1.0, e.task_fit * task_fit * self._task_multiplier(e, task_type)))
            reliability = max(0.05, min(1.0, e.reliability))
            quota = max(0.05, min(1.0, e.quota_remaining))
            speed = self._latency_score(e.latency_ms)
            return (fit ** 2) * (reliability ** 2) * (0.65 + 0.35 * quota) * (0.75 + 0.25 * speed)

        return max(candidates, key=score)

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
