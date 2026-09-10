from __future__ import annotations

import json
from typing import Any


class RedisTaskQueue:
    """Redis queue with optional processing-list recovery for worker crashes."""

    READY_KEY = "agent:tasks"
    PROCESSING_KEY = "agent:tasks:processing"

    def __init__(self, url: str):
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError("Install the redis extra: pip install '.[redis]'") from exc
        self.client = redis.Redis.from_url(url, decode_responses=True)

    def enqueue(self, task_id: str) -> None:
        self.client.lpush(self.READY_KEY, task_id)

    def dequeue(self, timeout: int = 5) -> str | None:
        item = self.client.brpop(self.READY_KEY, timeout=max(0, timeout))
        return item[1] if item else None

    def claim_reliable(self, timeout: int = 5) -> str | None:
        """Atomically move a task to processing; call ack() after completion."""
        item = self.client.brpoplpush(self.READY_KEY, self.PROCESSING_KEY, timeout=max(0, timeout))
        return item

    def ack(self, task_id: str) -> bool:
        return bool(self.client.lrem(self.PROCESSING_KEY, 1, task_id))

    def requeue(self, task_id: str) -> bool:
        removed = self.ack(task_id)
        if removed:
            self.enqueue(task_id)
        return removed

    def recover_processing(self, limit: int = 1000) -> list[str]:
        """Requeue tasks left in processing after a crashed worker."""
        limit = max(1, min(10000, int(limit)))
        items = self.client.lrange(self.PROCESSING_KEY, 0, limit - 1)
        if not items:
            return []
        pipe = self.client.pipeline()
        for task_id in items:
            pipe.lrem(self.PROCESSING_KEY, 1, task_id)
            pipe.lpush(self.READY_KEY, task_id)
        pipe.execute()
        return items

    def publish_event(self, task_id: str, event: dict[str, Any]) -> None:
        self.client.publish(f"agent:task:{task_id}", json.dumps(event, default=str))

    def ping(self) -> bool:
        return bool(self.client.ping())
