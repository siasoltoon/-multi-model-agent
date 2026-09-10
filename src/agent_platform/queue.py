from __future__ import annotations

import json
from typing import Any


class RedisTaskQueue:
    """Optional Redis queue for multi-worker deployments; task state remains in the database."""

    def __init__(self, url: str):
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError("Install the redis extra: pip install '.[redis]'") from exc
        self.client = redis.Redis.from_url(url, decode_responses=True)

    def enqueue(self, task_id: str) -> None:
        self.client.lpush("agent:tasks", task_id)

    def dequeue(self, timeout: int = 5) -> str | None:
        item = self.client.brpop("agent:tasks", timeout=timeout)
        return item[1] if item else None

    def publish_event(self, task_id: str, event: dict[str, Any]) -> None:
        self.client.publish(f"agent:task:{task_id}", json.dumps(event, default=str))

    def ping(self) -> bool:
        return bool(self.client.ping())
