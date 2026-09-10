from __future__ import annotations

import json
from uuid import UUID

from .models import Task, TaskStatus


class PostgresTaskStore:
    """Production PostgreSQL repository with reconnect-on-transient-failure behavior."""

    kind = "postgres"

    def __init__(self, dsn: str):
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError("Install the postgres extra: pip install '.[postgres]'") from exc
        self._psycopg = psycopg
        self.dsn = dsn
        self.db = psycopg.connect(dsn, autocommit=True)
        self._ensure_schema()

    def _connect(self) -> None:
        if getattr(self.db, "closed", False):
            self.db = self._psycopg.connect(self.dsn, autocommit=True)

    def _ensure_schema(self) -> None:
        self._connect()
        with self.db.cursor() as cur:
            cur.execute("CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, payload JSONB NOT NULL, updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW())")
            cur.execute("CREATE INDEX IF NOT EXISTS tasks_updated_idx ON tasks(updated_at DESC)")
            cur.execute("CREATE TABLE IF NOT EXISTS task_events (id BIGSERIAL PRIMARY KEY, task_id TEXT NOT NULL, event TEXT NOT NULL, payload JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW())")
            cur.execute("CREATE INDEX IF NOT EXISTS task_events_task_idx ON task_events(task_id, id DESC)")

    def _retry(self, operation):
        self._connect()
        try:
            return operation()
        except (self._psycopg.OperationalError, self._psycopg.InterfaceError):
            try:
                self.db.close()
            except Exception:
                pass
            self.db = self._psycopg.connect(self.dsn, autocommit=True)
            self._ensure_schema()
            return operation()

    def save(self, task: Task) -> Task:
        def op():
            with self.db.cursor() as cur:
                cur.execute("INSERT INTO tasks(id,payload,updated_at) VALUES(%s,%s::jsonb,NOW()) ON CONFLICT(id) DO UPDATE SET payload=EXCLUDED.payload,updated_at=NOW()", (str(task.id), task.model_dump_json()))
            return task
        return self._retry(op)

    def get(self, task_id: UUID) -> Task | None:
        def op():
            with self.db.cursor() as cur:
                cur.execute("SELECT payload::text FROM tasks WHERE id=%s", (str(task_id),))
                row = cur.fetchone()
            return Task.model_validate_json(row[0]) if row else None
        return self._retry(op)

    def list(self) -> list[Task]:
        def op():
            with self.db.cursor() as cur:
                cur.execute("SELECT payload::text FROM tasks ORDER BY updated_at DESC")
                rows = cur.fetchall()
            return [Task.model_validate_json(r[0]) for r in rows]
        return self._retry(op)

    def checkpoint(self, task_id: UUID, step: int, state: dict) -> Task:
        task = self.get(task_id)
        if not task:
            raise KeyError(task_id)
        task.current_step = step
        task.checkpoint = state
        task.status = TaskStatus.CHECKPOINTED
        saved = self.save(task)
        self.event(task.id, "checkpoint", {"step": step, "state": state})
        return saved

    def event(self, task_id: UUID, event: str, payload: dict) -> None:
        def op():
            with self.db.cursor() as cur:
                cur.execute("INSERT INTO task_events(task_id,event,payload) VALUES(%s,%s,%s::jsonb)", (str(task_id), event, json.dumps(payload, default=str)))
        self._retry(op)

    def events(self, task_id: UUID, limit: int = 200) -> list[dict]:
        limit = max(1, min(1000, int(limit)))
        def op():
            with self.db.cursor() as cur:
                cur.execute("SELECT event,payload::text,created_at FROM task_events WHERE task_id=%s ORDER BY id DESC LIMIT %s", (str(task_id), limit))
                rows = cur.fetchall()
            return [{"event": e, "payload": json.loads(p), "created_at": str(ts)} for e, p, ts in rows]
        return self._retry(op)

    def close(self) -> None:
        if not getattr(self.db, "closed", False):
            self.db.close()
