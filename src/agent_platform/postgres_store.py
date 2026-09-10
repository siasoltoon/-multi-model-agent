from __future__ import annotations

from uuid import UUID

from .models import Task


class PostgresTaskStore:
    """Production PostgreSQL repository. Install the optional postgres extra to use it."""

    def __init__(self, dsn: str):
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError("Install the postgres extra: pip install '.[postgres]'") from exc
        self.db = psycopg.connect(dsn, autocommit=True)
        with self.db.cursor() as cur:
            cur.execute("CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, payload JSONB NOT NULL, updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW())")
            cur.execute("CREATE INDEX IF NOT EXISTS tasks_updated_idx ON tasks(updated_at DESC)")
            cur.execute("CREATE TABLE IF NOT EXISTS task_events (id BIGSERIAL PRIMARY KEY, task_id TEXT NOT NULL, event TEXT NOT NULL, payload JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW())")

    def save(self, task: Task) -> Task:
        with self.db.cursor() as cur:
            cur.execute("INSERT INTO tasks(id,payload,updated_at) VALUES(%s,%s::jsonb,NOW()) ON CONFLICT(id) DO UPDATE SET payload=EXCLUDED.payload,updated_at=NOW()", (str(task.id), task.model_dump_json()))
        return task

    def get(self, task_id: UUID) -> Task | None:
        with self.db.cursor() as cur:
            cur.execute("SELECT payload::text FROM tasks WHERE id=%s", (str(task_id),))
            row = cur.fetchone()
        return Task.model_validate_json(row[0]) if row else None

    def list(self) -> list[Task]:
        with self.db.cursor() as cur:
            cur.execute("SELECT payload::text FROM tasks ORDER BY updated_at DESC")
            rows = cur.fetchall()
        return [Task.model_validate_json(r[0]) for r in rows]

    def checkpoint(self, task_id: UUID, step: int, state: dict) -> Task:
        task = self.get(task_id)
        if not task:
            raise KeyError(task_id)
        task.current_step = step
        task.checkpoint = state
        return self.save(task)

    def event(self, task_id: UUID, event: str, payload: dict) -> None:
        import json
        with self.db.cursor() as cur:
            cur.execute("INSERT INTO task_events(task_id,event,payload) VALUES(%s,%s,%s::jsonb)", (str(task_id), event, json.dumps(payload, default=str)))
