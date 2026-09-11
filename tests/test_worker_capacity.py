from types import SimpleNamespace

import pytest

from agent_platform.models import Task, TaskStatus
from agent_platform.worker_orchestrator import WorkerOrchestrator
from agent_platform.worker_protocol import WorkerProtocol
from agent_platform.workers import Worker, WorkerRegistry, WorkerStatus


def test_registry_reconciles_capacity_from_running_tasks():
    registry = WorkerRegistry()
    worker = registry.register(Worker(worker_id="laptop-01", max_concurrent_tasks=2))
    task = Task(prompt="run", status=TaskStatus.RUNNING, worker_id="laptop-01")

    registry.reconcile_active_tasks([task])

    assert worker.active_tasks == 1
    assert worker.available_slots == 1
    assert worker.status == WorkerStatus.BUSY


def test_orchestrator_reconciles_durable_load_before_selection(monkeypatch):
    registry = WorkerRegistry()
    worker = registry.register(Worker(worker_id="laptop-01", max_concurrent_tasks=1))
    running = Task(prompt="run", status=TaskStatus.RUNNING, worker_id="laptop-01")
    queued = Task(prompt="queued", status=TaskStatus.QUEUED)
    fake_store = SimpleNamespace(list=lambda: [running, queued])
    fake_app = SimpleNamespace(store=fake_store)
    monkeypatch.setitem(__import__("sys").modules, "agent_platform.app", fake_app)

    selected = WorkerOrchestrator(registry, github_enabled=False).select()

    assert selected is None
    assert worker.active_tasks == 1


def test_local_lease_rejects_duplicate_task_atomically():
    protocol = WorkerProtocol()
    protocol.issue("worker-a", "task-1", "lease-a", 300)

    with pytest.raises(RuntimeError, match="already leased"):
        protocol.issue("worker-b", "task-1", "lease-b", 300)

    assert protocol.valid("lease-a", "worker-a", "task-1")
    assert not protocol.valid("lease-b", "worker-b", "task-1")


def test_local_lease_allows_task_after_revoke():
    protocol = WorkerProtocol()
    protocol.issue("worker-a", "task-1", "lease-a", 300)
    protocol.revoke("lease-a")

    lease = protocol.issue("worker-b", "task-1", "lease-b", 300)

    assert lease.worker_id == "worker-b"
    assert protocol.valid("lease-b", "worker-b", "task-1")
