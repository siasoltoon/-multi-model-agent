from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone


@dataclass
class WorkerLease:
    worker_id: str
    task_id: str
    lease_id: str
    expires_at: datetime
    metadata: dict = field(default_factory=dict)


@dataclass
class WorkerProtocol:
    leases: dict[str, WorkerLease] = field(default_factory=dict)

    def issue(self, worker_id: str, task_id: str, lease_id: str, ttl_seconds: int = 300) -> WorkerLease:
        now = datetime.now(timezone.utc)
        lease = WorkerLease(worker_id, task_id, lease_id, now + timedelta(seconds=max(1, ttl_seconds)))
        self.leases[lease_id] = lease
        return lease

    def valid(self, lease_id: str, worker_id: str, task_id: str) -> bool:
        lease = self.leases.get(lease_id)
        return bool(lease and lease.worker_id == worker_id and lease.task_id == task_id and lease.expires_at > datetime.now(timezone.utc))

    def renew(self, lease_id: str, worker_id_or_ttl: str | int, ttl_seconds: int = 300) -> WorkerLease:
        lease = self.leases.get(lease_id)
        if isinstance(worker_id_or_ttl, int):
            worker_id = lease.worker_id if lease else ""
            ttl_seconds = worker_id_or_ttl
        else:
            worker_id = worker_id_or_ttl
        if not lease or lease.worker_id != worker_id or lease.expires_at <= datetime.now(timezone.utc):
            raise KeyError("lease expired or not found")
        lease.expires_at = datetime.now(timezone.utc) + timedelta(seconds=max(1, ttl_seconds))
        return lease

    def revoke(self, lease_id: str) -> None:
        self.leases.pop(lease_id, None)

    def reap_expired(self) -> list[str]:
        now = datetime.now(timezone.utc)
        expired = [key for key, lease in self.leases.items() if lease.expires_at <= now]
        for key in expired:
            self.leases.pop(key, None)
        return expired
