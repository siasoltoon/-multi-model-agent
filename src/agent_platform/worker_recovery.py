from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

from .models import TaskStatus
from .worker_protocol import WorkerProtocol
from .workers import WorkerRegistry


def recover_stale_laptop_tasks(
    store,
    workers: WorkerRegistry,
    leases,
    local_leases: WorkerProtocol,
    *,
    stale_seconds: int = 90,
    now: datetime | None = None,
    dispatch: Callable[[object], dict] | None = None,
) -> list[str]:
    """Requeue laptop tasks whose worker heartbeat/lease has expired.

    The function is deliberately side-effect-light so the control-plane
    lifecycle can call it periodically and tests can exercise recovery without
    starting FastAPI. A recovered task is returned to QUEUED state and can be
    claimed by a different laptop or routed to GitHub Actions on the next
    dispatch cycle.
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now.timestamp() - max(1, int(stale_seconds))
    stale_workers = set(workers.mark_stale(stale_seconds))
    recovered: list[str] = []

    for task in store.list():
        if task.status != TaskStatus.RUNNING:
            continue
        if str(task.metadata.get("worker_kind") or "") != "laptop":
            continue
        worker_id = str(task.worker_id or task.metadata.get("assigned_worker_id") or "")
        heartbeat_at = task.metadata.get("worker_heartbeat_at")
        heartbeat_ts = float(heartbeat_at) if heartbeat_at is not None else 0.0
        stale = worker_id in stale_workers or heartbeat_ts <= cutoff
        if not stale:
            continue

        lease_id = str(task.metadata.get("lease_id") or "")
        if lease_id:
            try:
                if leases:
                    leases.release_task(str(task.id))
                else:
                    local_leases.revoke(lease_id)
            except Exception:
                # Recovery must still make the task claimable if lease cleanup
                # temporarily fails; persistent lease expiry is the backstop.
                pass

        task.status = TaskStatus.QUEUED
        task.worker_id = None
        task.worker_run_id = None
        task.error = "worker became stale; task requeued for failover"
        task.metadata.pop("lease_id", None)
        task.metadata.pop("assigned_worker_id", None)
        task.metadata["worker_kind"] = ""
        task.metadata["recovered_at"] = now.timestamp()
        task.metadata["recovery_reason"] = "worker_stale"
        store.save(task)
        if hasattr(store, "event"):
            store.event(task.id, "worker_recovered", {"worker_id": worker_id, "reason": "worker_stale"})
        recovered.append(str(task.id))

        if dispatch:
            try:
                dispatch(task)
            except Exception as exc:
                task.status = TaskStatus.QUEUED
                task.error = str(exc)
                store.save(task)

    return recovered
