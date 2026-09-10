from __future__ import annotations

from dataclasses import dataclass


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


class SmartRouter:
    def __init__(self, endpoints: list[ModelEndpoint] | None = None):
        self.endpoints = endpoints or []

    def register(self, endpoint: ModelEndpoint) -> None:
        self.endpoints = [x for x in self.endpoints if x.id != endpoint.id]
        self.endpoints.append(endpoint)

    def choose(self, *, task_fit: float = 1.0, min_context: int = 0, tools: bool = False) -> ModelEndpoint:
        candidates = [e for e in self.endpoints if e.health not in {"OFFLINE", "RATE_LIMITED", "QUOTA_EXHAUSTED"}
                      and e.context_window >= min_context and (not tools or e.tool_support)]
        if not candidates:
            raise RuntimeError("no healthy model endpoint available")
        return max(candidates, key=lambda e: (e.task_fit * task_fit) * e.reliability * max(e.quota_remaining, 0.01) / max(e.latency_ms, 1))

    def update_health(self, endpoint_id: str, health: str, *, latency_ms: float | None = None, quota_remaining: float | None = None) -> None:
        for e in self.endpoints:
            if e.id == endpoint_id:
                e.health = health
                if latency_ms is not None:
                    e.latency_ms = latency_ms
                if quota_remaining is not None:
                    e.quota_remaining = max(0.0, quota_remaining)
