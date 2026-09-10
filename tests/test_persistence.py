from uuid import uuid4
from agent_platform.models import Task, TaskStatus
from agent_platform.persistent_store import PersistentTaskStore
from agent_platform.worker_protocol import WorkerProtocol

def test_task_store_survives_new_instance(tmp_path):
    url = f"sqlite:///{tmp_path / 'tasks.db'}"
    first = PersistentTaskStore(url)
    task = first.save(Task(prompt="persist me"))
    second = PersistentTaskStore(url)
    loaded = second.get(task.id)
    assert loaded is not None and loaded.prompt == "persist me"

def test_checkpoint_is_resumable(tmp_path):
    store = PersistentTaskStore(f"sqlite:///{tmp_path / 'tasks.db'}")
    task = store.save(Task(prompt="checkpoint"))
    saved = store.checkpoint(task.id, 12, {"node": "implement"})
    assert saved.status == TaskStatus.CHECKPOINTED and saved.current_step == 12

def test_worker_lease_expires():
    protocol = WorkerProtocol()
    lease = protocol.issue("w1", "t1", "l1", ttl_seconds=60)
    assert protocol.valid(lease.lease_id, "w1", "t1")
    protocol.revoke(lease.lease_id)
    assert not protocol.valid("l1", "w1", "t1")
