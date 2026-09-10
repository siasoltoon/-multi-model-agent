import json
import sqlite3
from pathlib import Path
from threading import Lock
from uuid import UUID

from .models import Task, TaskStatus


class PersistentTaskStore:
    """Small durable SQLite store; replaceable by PostgreSQL without changing the API layer."""
    def __init__(self, url: str = "sqlite:///./agent.db"):
        path = url.removeprefix("sqlite:///")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = Lock()
        self._init()

    def _connect(self):
        return sqlite3.connect(self.path, check_same_thread=False)

    def _init(self):
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")

    def save(self, task: Task) -> Task:
        with self.lock, self._connect() as db:
            db.execute("INSERT OR REPLACE INTO tasks(id,payload) VALUES (?,?)", (str(task.id), task.model_dump_json()))
        return task.model_copy(deep=True)

    def get(self, task_id: UUID) -> Task | None:
        with self.lock, self._connect() as db:
            row = db.execute("SELECT payload FROM tasks WHERE id=?", (str(task_id),)).fetchone()
        return Task.model_validate_json(row[0]) if row else None

    def list(self) -> list[Task]:
        with self.lock, self._connect() as db:
            rows = db.execute("SELECT payload FROM tasks ORDER BY rowid DESC").fetchall()
        return [Task.model_validate_json(r[0]) for r in rows]

    def checkpoint(self, task_id: UUID, step: int, data: dict) -> Task:
        task = self.get(task_id)
        if not task:
            raise KeyError(task_id)
        task.current_step = step
        task.checkpoint = {"step": step, "data": data}
        task.status = TaskStatus.CHECKPOINTED
        return self.save(task)
