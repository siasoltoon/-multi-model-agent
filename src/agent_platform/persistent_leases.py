from __future__ import annotations

from datetime import datetime, timedelta, timezone


class PersistentWorkerLeases:
    """PostgreSQL-backed leases for safe multi-worker task claiming and recovery."""

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
            cur.execute("CREATE TABLE IF NOT EXISTS worker_leases (task_id TEXT PRIMARY KEY, worker_id TEXT NOT NULL, lease_id TEXT NOT NULL UNIQUE, expires_at TIMESTAMPTZ NOT NULL, updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW())")
            cur.execute("CREATE INDEX IF NOT EXISTS worker_leases_expiry_idx ON worker_leases(expires_at)")

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

    def claim(self, worker_id: str, task_id: str, lease_id: str, ttl_seconds: int = 300) -> bool:
        expires = datetime.now(timezone.utc) + timedelta(seconds=max(1, ttl_seconds))
        def op():
            with self.db.cursor() as cur:
                cur.execute("INSERT INTO worker_leases(task_id,worker_id,lease_id,expires_at) VALUES(%s,%s,%s,%s) ON CONFLICT(task_id) DO UPDATE SET worker_id=EXCLUDED.worker_id,lease_id=EXCLUDED.lease_id,expires_at=EXCLUDED.expires_at,updated_at=NOW() WHERE worker_leases.expires_at <= NOW()", (task_id, worker_id, lease_id, expires))
                return cur.rowcount == 1
        return self._retry(op)

    def renew(self, worker_id: str, lease_id: str, ttl_seconds: int = 300) -> bool:
        expires = datetime.now(timezone.utc) + timedelta(seconds=max(1, ttl_seconds))
        def op():
            with self.db.cursor() as cur:
                cur.execute("UPDATE worker_leases SET expires_at=%s,updated_at=NOW() WHERE worker_id=%s AND lease_id=%s AND expires_at>NOW()", (expires, worker_id, lease_id))
                return cur.rowcount == 1
        return self._retry(op)

    def valid(self, worker_id: str, task_id: str, lease_id: str) -> bool:
        def op():
            with self.db.cursor() as cur:
                cur.execute("SELECT 1 FROM worker_leases WHERE worker_id=%s AND task_id=%s AND lease_id=%s AND expires_at>NOW()", (worker_id, task_id, lease_id))
                return cur.fetchone() is not None
        return self._retry(op)

    def release(self, worker_id: str, lease_id: str) -> bool:
        def op():
            with self.db.cursor() as cur:
                cur.execute("DELETE FROM worker_leases WHERE worker_id=%s AND lease_id=%s", (worker_id, lease_id))
                return cur.rowcount == 1
        return self._retry(op)

    def reap_expired(self, limit: int = 1000) -> int:
        """Remove expired leases so abandoned work can be claimed again."""
        limit = max(1, min(10000, int(limit)))
        def op():
            with self.db.cursor() as cur:
                cur.execute("DELETE FROM worker_leases WHERE task_id IN (SELECT task_id FROM worker_leases WHERE expires_at <= NOW() ORDER BY expires_at LIMIT %s)", (limit,))
                return cur.rowcount
        return self._retry(op)

    def release_task(self, task_id: str) -> bool:
        """Release a task lease without trusting a stale worker identity."""
        def op():
            with self.db.cursor() as cur:
                cur.execute("DELETE FROM worker_leases WHERE task_id=%s", (task_id,))
                return cur.rowcount == 1
        return self._retry(op)

    def close(self) -> None:
        if not getattr(self.db, "closed", False):
            self.db.close()
