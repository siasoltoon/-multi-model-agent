from __future__ import annotations

import json
import sqlite3
import threading
from uuid import UUID

from .models import Task, TaskStatus


class PersistentTaskStore:
    """SQLite durable repository used for development and single-instance deployments."""

    kind = "sqlite"

    def __init__(self, url: str):
        if not url.startswith("sqlite:///"):
            raise ValueError("use PostgresTaskStore for postgresql:// URLs")
        self.db = sqlite3.connect(
            url.removeprefix("sqlite:///"),
            check_same_thread=False,
            timeout=30,
        )
        self._lock = threading.RLock()
        self.db.execute("PRAGMA busy_timeout=30000")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, payload TEXT NOT NULL, updated_at TEXT DEFAULT CURRENT_TIMESTAMP)")
        self.db.execute("CREATE INDEX IF NOT EXISTS tasks_updated_idx ON tasks(updated_at DESC)")
        self.db.execute("CREATE TABLE IF NOT EXISTS task_events (id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL, event TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP)")
        self.db.execute("CREATE INDEX IF NOT EXISTS task_events_task_idx ON task_events(task_id, id DESC)")
        self.db.commit()

    def save(self, task: Task) -> Task:
        with self._lock:
            self.db.execute("INSERT OR REPLACE INTO tasks(id,payload,updated_at) VALUES(?,?,CURRENT_TIMESTAMP)", (str(task.id), task.model_dump_json()))
            self.db.commit()
        return task

    def get(self, task_id: UUID) -> Task | None:
        with self._lock:
            row = self.db.execute("SELECT payload FROM tasks WHERE id=?", (str(task_id),)).fetchone()
        return Task.model_validate_json(row[0]) if row else None

    def list(self) -> list[Task]:
        with self._lock:
            rows = self.db.execute("SELECT payload FROM tasks ORDER BY updated_at DESC").fetchall()
        return [Task.model_validate_json(row[0]) for row in rows]

    def checkpoint(self, task_id: UUID, step: int, state: dict) -> Task:
        task = self.get(task_id)
        if not task:
            raise KeyError(task_id)
        task.current_step = step; task.checkpoint = state; task.status = TaskStatus.CHECKPOINTED
        return self.save(task)

    def event(self, task_id: UUID, event: str, payload: dict) -> None:
        with self._lock:
            self.db.execute("INSERT INTO task_events(task_id,event,payload) VALUES(?,?,?)", (str(task_id), event, json.dumps(payload, ensure_ascii=False, default=str)))
            self.db.commit()

    def events(self, task_id: UUID, limit: int = 200) -> list[dict]:
        limit = max(1, min(1000, int(limit)))
        with self._lock:
            rows = self.db.execute("SELECT event,payload,created_at FROM task_events WHERE task_id=? ORDER BY id DESC LIMIT ?", (str(task_id), limit)).fetchall()
        return [{"event": e, "payload": json.loads(p), "created_at": str(ts)} for e, p, ts in rows]

    def close(self) -> None:
        with self._lock:
            self.db.close()
