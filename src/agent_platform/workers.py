from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from pydantic import BaseModel, Field


class WorkerStatus(str, Enum):
    ONLINE = "online"
    BUSY = "busy"
    OFFLINE = "offline"


class Worker(BaseModel):
    worker_id: str
    endpoint: str = ""
    kind: str = "laptop"
    capabilities: list[str] = Field(default_factory=lambda: ["coding", "testing", "tools"])
    cpu_cores: int | None = None
    memory_mb: int | None = None
    gpu: bool = False
    models: list[str] = Field(default_factory=list)
    max_concurrent_tasks: int = Field(default=1, ge=1, le=256)
    active_tasks: int = Field(default=0, ge=0, le=256)
    status: WorkerStatus = WorkerStatus.ONLINE
    last_heartbeat: datetime | None = None

    def capability_set(self) -> set[str]:
        return {str(item).strip().lower() for item in self.capabilities if str(item).strip()}

    @property
    def available_slots(self) -> int:
        return max(0, self.max_concurrent_tasks - self.active_tasks)

    @property
    def load_ratio(self) -> float:
        return min(1.0, self.active_tasks / max(1, self.max_concurrent_tasks))


class WorkerRegistry:
    def __init__(self):
        self.workers: dict[str, Worker] = {}

    def register(self, worker: Worker):
        worker.last_heartbeat = datetime.now(timezone.utc)
        worker.active_tasks = 0
        worker.status = WorkerStatus.ONLINE
        self.workers[worker.worker_id] = worker
        return worker

    def heartbeat(self, worker_id: str, status: WorkerStatus | None = None):
        if worker_id not in self.workers:
            raise KeyError(worker_id)
        w = self.workers[worker_id]
        w.last_heartbeat = datetime.now(timezone.utc)
        if status is not None:
            if status == WorkerStatus.ONLINE and w.active_tasks > 0:
                w.status = WorkerStatus.BUSY
            else:
                w.status = status
        return w

    def claim(self, worker_id: str) -> Worker:
        if worker_id not in self.workers:
            raise KeyError(worker_id)
        w = self.workers[worker_id]
        if w.status == WorkerStatus.OFFLINE:
            raise RuntimeError(f"worker {worker_id} is offline")
        if w.available_slots <= 0:
            raise RuntimeError(f"worker {worker_id} has no capacity")
        w.active_tasks += 1
        w.status = WorkerStatus.BUSY
        return w

    def release(self, worker_id: str) -> Worker:
        if worker_id not in self.workers:
            raise KeyError(worker_id)
        w = self.workers[worker_id]
        w.active_tasks = max(0, w.active_tasks - 1)
        if w.status != WorkerStatus.OFFLINE:
            w.status = WorkerStatus.BUSY if w.active_tasks else WorkerStatus.ONLINE
        w.last_heartbeat = datetime.now(timezone.utc)
        return w

    def mark_stale(self, timeout_seconds: int = 90) -> list[str]:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=max(1, timeout_seconds))
        stale: list[str] = []
        for worker_id, worker in self.workers.items():
            if worker.last_heartbeat is None or worker.last_heartbeat < cutoff:
                worker.status = WorkerStatus.OFFLINE
                stale.append(worker_id)
        return stale

    def online(self, timeout_seconds: int = 90) -> list[Worker]:
        self.mark_stale(timeout_seconds)
        return [w for w in self.workers.values() if w.status != WorkerStatus.OFFLINE]
