from uuid import uuid4

from agent_platform.models import Task, TaskStatus
from agent_platform.persistent_store import PersistentTaskStore


def test_task_store_survives_new_instance(tmp_path):
    url = f"sqlite:///{tmp_path / 'tasks.db'}"
    first = PersistentTaskStore(url)
    task = first.save(Task(id=uuid4(), prompt="persist me"))
    second = PersistentTaskStore(url)
    loaded = second.get(task.id)
    assert loaded is not None
    assert loaded.prompt == "persist me"


def test_checkpoint_is_resumable(tmp_path):
    store = PersistentTaskStore(f"sqlite:///{tmp_path / 'tasks.db'}")
    task = store.save(Task(id=uuid4(), prompt="checkpoint"))
    saved = store.checkpoint(task.id, 12, {"node": "implement"})
    assert saved.status == TaskStatus.CHECKPOINTED
    assert saved.current_step == 12
