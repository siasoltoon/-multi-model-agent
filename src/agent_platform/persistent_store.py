from __future__ import annotations
import json
import sqlite3
from uuid import UUID
from .models import Task, TaskStatus

class PersistentTaskStore:
    def __init__(self, url: str):
        if not url.startswith("sqlite:///"): raise ValueError("only sqlite is supported by the default store")
        self.path = url.removeprefix("sqlite:///")
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        self.db.commit()
    def save(self, task: Task) -> Task:
        self.db.execute("INSERT OR REPLACE INTO tasks(id,payload) VALUES(?,?)", (str(task.id), task.model_dump_json()))
        self.db.commit(); return task
    def get(self, task_id: UUID) -> Task | None:
        row = self.db.execute("SELECT payload FROM tasks WHERE id=?", (str(task_id),)).fetchone()
        return Task.model_validate_json(row[0]) if row else None
    def list(self) -> list[Task]:
        return [Task.model_validate_json(row[0]) for row in self.db.execute("SELECT payload FROM tasks ORDER BY rowid DESC")]
    def checkpoint(self, task_id: UUID, step: int, state: dict) -> Task:
        task = self.get(task_id)
        if not task: raise KeyError(task_id)
        task.current_step = step; task.checkpoint = state; task.status = TaskStatus.CHECKPOINTED
        return self.save(task)
