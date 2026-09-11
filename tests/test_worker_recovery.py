from datetime import datetime, timedelta, timezone
from uuid import uuid4

from agent_platform.models import Task, TaskStatus
from agent_platform.worker_protocol import WorkerProtocol
from agent_platform.worker_recovery import recover_stale_laptop_tasks
from agent_platform.workers import Worker, WorkerRegistry


class FakeStore:
    def __init__(self, tasks):
        self.tasks = {str(task.id): task for task in tasks}
        self.events = []

    def list(self):
        return list(self.tasks.values())

    def save(self, task):
        self.tasks[str(task.id)] = task
        return task

    def event(self, task_id, event, payload):
        self.events.append((str(task_id), event, payload))


def test_stale_laptop_task_is_requeued_and_lease_released():
    registry = WorkerRegistry()
    worker = registry.register(Worker(worker_id="laptop-01"))
    worker.last_heartbeat = datetime.now(timezone.utc) - timedelta(minutes=5)
    task = Task(status=TaskStatus.RUNNING, worker_id="laptop-01", metadata={"worker_kind": "laptop", "lease_id": "lease-1"})
    store = FakeStore([task])
    leases = WorkerProtocol()
    leases.issue("laptop-01", str(task.id), "lease-1", ttl_seconds=300)

    recovered = recover_stale_laptop_tasks(store, registry, None, leases, stale_seconds=90)

    assert recovered == [str(task.id)]
    assert task.status == TaskStatus.QUEUED
    assert task.worker_id is None
    assert "lease_id" not in task.metadata
    assert not leases.valid("laptop-01", str(task.id), "lease-1")
    assert any(event[1] == "worker_recovered" for event in store.events)


def test_healthy_laptop_task_is_not_recovered():
    registry = WorkerRegistry()
    registry.register(Worker(worker_id="laptop-01"))
    task = Task(status=TaskStatus.RUNNING, worker_id="laptop-01", metadata={"worker_kind": "laptop", "worker_heartbeat_at": datetime.now(timezone.utc).timestamp()})
    store = FakeStore([task])

    recovered = recover_stale_laptop_tasks(store, registry, None, WorkerProtocol(), stale_seconds=90)

    assert recovered == []
    assert task.status == TaskStatus.RUNNING


def test_non_laptop_task_is_not_recovered():
    registry = WorkerRegistry()
    task = Task(status=TaskStatus.RUNNING, worker_id="github-actions", metadata={"worker_kind": "github-actions"})
    store = FakeStore([task])

    recovered = recover_stale_laptop_tasks(store, registry, None, WorkerProtocol(), stale_seconds=90)

    assert recovered == []
    assert task.status == TaskStatus.RUNNING
