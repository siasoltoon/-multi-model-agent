from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class WorkerLease:
    worker_id: str
    task_id: str
    lease_id: str
    expires_at: datetime


@dataclass
class WorkerProtocol:
    leases: dict[str, WorkerLease] = field(default_factory=dict)

    def issue(self, worker_id: str, task_id: str, lease_id: str, ttl_seconds: int = 300) -> WorkerLease:
        lease = WorkerLease(worker_id, task_id, lease_id, datetime.now(timezone.utc).replace(microsecond=0))
        self.leases[lease_id] = lease
        return lease

    def valid(self, lease_id: str, worker_id: str, task_id: str) -> bool:
        lease = self.leases.get(lease_id)
        return bool(lease and lease.worker_id == worker_id and lease.task_id == task_id)

    def revoke(self, lease_id: str) -> None:
        self.leases.pop(lease_id, None)
