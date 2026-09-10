from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import httpx

from .discovery import env_api_key
from .router import ModelEndpoint, SmartRouter


@dataclass
class HealthProbeResult:
    endpoint_id: str
    health: str
    latency_ms: float
    quota_remaining: float | None = None
    status_code: int | None = None
    error: str | None = None


class ProviderHealthMonitor:
    """Probe provider endpoints and feed availability, latency and quota into the router."""

    def __init__(self, router: SmartRouter, timeout: float = 10.0):
        self.router = router
        self.timeout = timeout

    @staticmethod
    def _headers(endpoint: ModelEndpoint) -> dict[str, str]:
        key = env_api_key(endpoint.api_key_env)
        return {"Authorization": f"Bearer {key}"} if key else {}

    @staticmethod
    def _quota(headers: httpx.Headers) -> float | None:
        remaining = headers.get("x-ratelimit-remaining") or headers.get("x-rate-limit-remaining")
        limit = headers.get("x-ratelimit-limit") or headers.get("x-rate-limit-limit")
        if remaining is None or limit is None:
            return None
        try:
            limit_value = float(limit)
            if limit_value <= 0:
                return None
            return max(0.0, min(1.0, float(remaining) / limit_value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _probe_url(endpoint: ModelEndpoint) -> str:
        base = endpoint.base_url.rstrip("/")
        if endpoint.provider.lower() == "ollama":
            return base + "/api/tags"
        return base + "/models"

    async def probe(self, endpoint: ModelEndpoint) -> HealthProbeResult:
        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(self._probe_url(endpoint), headers=self._headers(endpoint))
            latency = (time.perf_counter() - started) * 1000.0
            quota = self._quota(response.headers)
            if response.status_code == 429:
                self.router.mark_failure(endpoint.id, "HTTP 429 Too Many Requests", rate_limited=True)
                return HealthProbeResult(endpoint.id, "RATE_LIMITED", latency, quota, response.status_code)
            if response.status_code in {401, 403}:
                self.router.update_health(endpoint.id, "AUTH_ERROR", latency_ms=latency, quota_remaining=quota)
                return HealthProbeResult(endpoint.id, "AUTH_ERROR", latency, quota, response.status_code)
            response.raise_for_status()
            health = "ONLINE" if (quota is None or quota > 0) else "QUOTA_EXHAUSTED"
            self.router.update_health(endpoint.id, health, latency_ms=latency, quota_remaining=1.0 if quota is None else quota)
            if health == "ONLINE":
                self.router.mark_success(endpoint.id, latency_ms=latency)
            return HealthProbeResult(endpoint.id, health, latency, quota, response.status_code)
        except httpx.TimeoutException as exc:
            latency = (time.perf_counter() - started) * 1000.0
            self.router.mark_failure(endpoint.id, exc)
            return HealthProbeResult(endpoint.id, "TIMEOUT", latency, error=str(exc))
        except Exception as exc:
            latency = (time.perf_counter() - started) * 1000.0
            self.router.mark_failure(endpoint.id, exc)
            return HealthProbeResult(endpoint.id, "ERROR", latency, error=str(exc))

    async def probe_all(self) -> list[HealthProbeResult]:
        results = []
        for endpoint in self.router.endpoints:
            results.append(await self.probe(endpoint))
        return results

    def snapshot(self) -> list[dict[str, Any]]:
        return [
            {
                "id": e.id,
                "provider": e.provider,
                "model": e.model,
                "health": e.health,
                "latency_ms": e.latency_ms,
                "quota_remaining": e.quota_remaining,
                "failures": e.failures,
                "cooldown_until": e.cooldown_until,
            }
            for e in self.router.endpoints
        ]
