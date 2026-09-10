from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel

class WorkerStatus(str, Enum):
    ONLINE = "online"
    BUSY = "busy"
    OFFLINE = "offline"

class Worker(BaseModel):
    worker_id: str
    endpoint: str = ""
    status: WorkerStatus = WorkerStatus.ONLINE
    last_heartbeat: datetime | None = None

class WorkerRegistry:
    def __init__(self): self.workers: dict[str, Worker] = {}
    def register(self, worker: Worker):
        worker.last_heartbeat = datetime.now(timezone.utc); self.workers[worker.worker_id] = worker; return worker
    def heartbeat(self, worker_id: str, status: WorkerStatus | None = None):
        if worker_id not in self.workers: raise KeyError(worker_id)
        w = self.workers[worker_id]; w.last_heartbeat = datetime.now(timezone.utc)
        if status is not None: w.status = status
        return w
    def claim(self, worker_id: str) -> Worker:
        if worker_id not in self.workers: raise KeyError(worker_id)
        w = self.workers[worker_id]; w.status = WorkerStatus.BUSY; return w
    def release(self, worker_id: str) -> Worker:
        if worker_id not in self.workers: raise KeyError(worker_id)
        w = self.workers[worker_id]; w.status = WorkerStatus.ONLINE; return w
