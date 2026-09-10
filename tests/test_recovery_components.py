from __future__ import annotations

from datetime import datetime, timedelta, timezone

from agent_platform.worker_protocol import WorkerProtocol
from agent_platform.workers import Worker, WorkerRegistry, WorkerStatus


def test_worker_registry_marks_expired_heartbeat_offline():
    registry = WorkerRegistry()
    worker = Worker(worker_id="w1")
    registry.register(worker)
    worker.last_heartbeat = datetime.now(timezone.utc) - timedelta(seconds=120)

    stale = registry.mark_stale(timeout_seconds=90)

    assert stale == ["w1"]
    assert worker.status == WorkerStatus.OFFLINE
    assert registry.online(timeout_seconds=90) == []


def test_worker_protocol_rejects_expired_and_wrong_owner_leases():
    protocol = WorkerProtocol()
    lease = protocol.issue("worker-a", "task-1", "lease-1", ttl_seconds=60)

    assert protocol.valid("lease-1", "worker-a", "task-1")
    assert not protocol.valid("lease-1", "worker-b", "task-1")
    assert not protocol.valid("lease-1", "worker-a", "task-2")

    protocol.revoke(lease.lease_id)
    assert not protocol.valid("lease-1", "worker-a", "task-1")


def test_worker_protocol_reaps_expired_leases():
    protocol = WorkerProtocol()
    lease = protocol.issue("worker-a", "task-1", "lease-1", ttl_seconds=1)
    lease.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)

    assert protocol.reap_expired() == ["lease-1"]
    assert not protocol.valid("lease-1", "worker-a", "task-1")
