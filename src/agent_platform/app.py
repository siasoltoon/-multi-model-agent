from __future__ import annotations

import asyncio
import json
import re
import time
from contextlib import suppress
from urllib.parse import urlparse
from uuid import UUID, uuid4

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .config import settings
from .discovery import ProviderDiscovery
from .engine import AgentEngine
from .github_dispatcher import GitHubActionsDispatcher
from .health import ProviderHealthMonitor
from .models import HealthResponse, Task, TaskRequest, TaskStatus
from .persistent_store import PersistentTaskStore
from .reliability import redact_secrets, verify_callback_signature
from .router import ModelEndpoint, SmartRouter
from .runner import run_task
from .worker_orchestrator import WorkerKind, WorkerOrchestrator
from .worker_protocol import WorkerProtocol
from .worker_recovery import recover_stale_laptop_tasks
from .workers import Worker, WorkerRegistry, WorkerStatus

RELEASE_VERSION = "1.0.0"
_REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_REF_RE = re.compile(r"^[A-Za-z0-9._/@-]{1,255}$")
_WORKFLOW_RE = re.compile(r"^[A-Za-z0-9._/-]+[.]ya?ml$")
_BRANCH_RE = re.compile(r"^[A-Za-z0-9._/@-]{1,255}$")


def _store():
    if settings.database_url.startswith(("postgresql://", "postgres://")):
        from .postgres_store import PostgresTaskStore
        return PostgresTaskStore(settings.database_url)
    return PersistentTaskStore(settings.database_url)


store = _store()
workers = WorkerRegistry()
router = SmartRouter()
engine = AgentEngine(router)
discovery = ProviderDiscovery()
health_monitor = ProviderHealthMonitor(router)
leases = None
if settings.database_url.startswith(("postgresql://", "postgres://")):
    from .persistent_leases import PersistentWorkerLeases
    leases = PersistentWorkerLeases(settings.database_url)
local_leases = WorkerProtocol()
orchestrator = WorkerOrchestrator(workers, github_enabled=True)
app = FastAPI(title=settings.app_name, version=RELEASE_VERSION)
_recovery_task: asyncio.Task | None = None


def _authorized(token: str | None) -> None:
    if settings.worker_auth_token and token != settings.worker_auth_token:
        raise HTTPException(401, "invalid worker token")


def _callback_authorized(token: str | None) -> None:
    expected = settings.github_callback_token or settings.worker_auth_token
    if expected and token != expected:
        raise HTTPException(401, "invalid callback token")


def _validate_github_target(repository: str, workflow: str, ref: str, branch: str) -> None:
    if not _REPOSITORY_RE.fullmatch(repository):
        raise HTTPException(400, "invalid GitHub repository; expected owner/name")
    if not _WORKFLOW_RE.fullmatch(workflow) or workflow.startswith("/") or ".." in workflow.split("/"):
        raise HTTPException(400, "invalid GitHub Actions workflow path")
    if not _REF_RE.fullmatch(ref) or ref.startswith("/") or ".." in ref.split("/"):
        raise HTTPException(400, "invalid GitHub Actions ref")
    if not _BRANCH_RE.fullmatch(branch) or branch.startswith("/") or ".." in branch.split("/"):
        raise HTTPException(400, "invalid worker branch")


def _validate_callback_url(value: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
        raise HTTPException(400, "callback_url must be an absolute HTTP(S) URL")
    if settings.public_base_url and parsed.scheme != "https" and settings.public_base_url.startswith("https://"):
        raise HTTPException(400, "callback_url must use HTTPS when AGENT_PUBLIC_BASE_URL uses HTTPS")


def _register(items):
    for item in items:
        metadata = dict(item.metadata or {})
        metadata["billing_type"] = item.billing_type
        router.register(ModelEndpoint(id=f"{item.provider}:{item.model}:{item.base_url}", provider=item.provider, model=item.model, base_url=item.base_url, context_window=item.context_window, tool_support=item.tool_support, task_fit=item.task_fit, reliability=item.reliability, latency_ms=item.latency_ms, quota_remaining=1.0, api_key_env=item.api_key_env, metadata=metadata))


def _recover_once() -> list[str]:
    return recover_stale_laptop_tasks(
        store,
        workers,
        leases,
        local_leases,
        stale_seconds=max(30, settings.worker_heartbeat_seconds * 6),
        dispatch=_dispatch,
    )


async def _recovery_loop() -> None:
    interval = max(10, settings.worker_heartbeat_seconds * 2)
    while True:
        try:
            _recover_once()
            if leases:
                with suppress(Exception):
                    leases.reap_expired()
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        await asyncio.sleep(interval)


@app.on_event("startup")
async def startup_lifecycle():
    global _recovery_task
    _register(await discovery.discover())
    workers.mark_stale(timeout_seconds=max(30, settings.worker_heartbeat_seconds * 6))
    if leases:
        with suppress(Exception):
            leases.reap_expired()
    if settings.redis_url:
        try:
            from .queue import RedisTaskQueue
            RedisTaskQueue(settings.redis_url).recover_processing_once()
        except Exception:
            pass
    _recovery_task = asyncio.create_task(_recovery_loop())


@app.on_event("shutdown")
async def shutdown_lifecycle():
    global _recovery_task
    if _recovery_task:
        _recovery_task.cancel()
        with suppress(asyncio.CancelledError):
            await _recovery_task
        _recovery_task = None
    for resource in (leases, store):
        close = getattr(resource, "close", None)
        if close:
            with suppress(Exception):
                close()


@app.get("/health", response_model=HealthResponse)
def health():
    redis_ok = None
    if settings.redis_url:
        try:
            from .queue import RedisTaskQueue
            redis_ok = RedisTaskQueue(settings.redis_url).ping()
        except Exception:
            redis_ok = False
    return {"status": "ok" if redis_ok is not False else "degraded", "database": "postgres" if leases else "sqlite", "redis": redis_ok}


@app.get("/")
def root():
    return JSONResponse({"name": settings.app_name, "version": RELEASE_VERSION, "interface": "terminal", "message": "Web UI is intentionally disabled. Use the multi-model-agent CLI.", "commands": ["multi-model-agent submit", "multi-model-agent worker", "multi-model-agent watch", "multi-model-agent status", "multi-model-agent plan", "multi-model-agent events", "multi-model-agent resume", "multi-model-agent cancel", "multi-model-agent run-local"]})


@app.post("/api/providers/discover")
async def discover_providers():
    items = await discovery.discover()
    _register(items)
    return {"count": len(items), "providers": sorted({x.provider for x in items}), "models": sorted({x.model for x in items})}


@app.post("/api/providers/health")
async def probe_provider_health():
    results = await health_monitor.probe_all()
    return {"count": len(results), "results": [result.__dict__ for result in results], "providers": health_monitor.snapshot()}


@app.get("/api/providers")
def providers():
    return [{"id": e.id, "provider": e.provider, "model": e.model, "base_url": e.base_url, "health": e.health, "billing_type": e.metadata.get("billing_type", "unknown"), "context_window": e.context_window, "tool_support": e.tool_support, "latency_ms": e.latency_ms, "quota_remaining": e.quota_remaining, "failures": e.failures} for e in router.endpoints]


@app.get("/api/providers/health")
def provider_health_snapshot():
    return health_monitor.snapshot()


@app.post("/api/tasks", response_model=Task)
def create_task(request: TaskRequest):
    task = Task(prompt=request.prompt, max_steps=request.max_steps or settings.max_agent_steps, metadata=request.metadata)
    store.save(task)
    if hasattr(store, "event"):
        store.event(task.id, "created", {"max_steps": task.max_steps})
    if settings.redis_url:
        try:
            from .queue import RedisTaskQueue
            RedisTaskQueue(settings.redis_url).enqueue(str(task.id))
        except Exception:
            pass
    return task


@app.get("/api/tasks", response_model=list[Task])
def list_tasks():
    return store.list()


@app.get("/api/tasks/{task_id}", response_model=Task)
def get_task(task_id: UUID):
    task = store.get(task_id)
    if not task:
        raise HTTPException(404, "task not found")
    return task


@app.get("/api/tasks/{task_id}/events")
def task_events(task_id: UUID):
    if not hasattr(store, "events"):
        raise HTTPException(501, "event listing unavailable")
    return store.events(task_id)


@app.post("/api/tasks/{task_id}/plan")
def plan_task(task_id: UUID):
    task = store.get(task_id)
    if not task:
        raise HTTPException(404, "task not found")
    plan = engine.plan(task)
    task.status = TaskStatus.PLANNING
    store.save(task)
    return {"task_id": str(task.id), "nodes": [{"id": n.id, "role": n.role, "dependencies": sorted(n.dependencies)} for n in plan.dag.nodes.values()]}


def _github_dispatch(task: Task) -> dict:
    repository = str(task.metadata.get("repository") or settings.github_worker_repository)
    if not repository:
        raise HTTPException(400, "repository is required in task metadata or AGENT_GITHUB_WORKER_REPOSITORY")
    workflow = str(task.metadata.get("worker_workflow") or settings.github_worker_workflow)
    ref = str(task.metadata.get("worker_ref") or settings.github_worker_ref)
    dispatcher = GitHubActionsDispatcher(settings.github_token)
    branch = str(task.metadata.get("worker_branch") or f"agent/task-{task.id}")
    callback_url = str(task.metadata.get("callback_url") or settings.public_base_url).rstrip("/")
    if not callback_url:
        raise HTTPException(400, "AGENT_PUBLIC_BASE_URL is required for automatic worker callbacks")
    _validate_github_target(repository, workflow, ref, branch)
    _validate_callback_url(callback_url)
    task.metadata.update({"repository": repository, "worker_branch": branch, "worker_workflow": workflow, "worker_ref": ref, "callback_url": callback_url, "worker_kind": WorkerKind.GITHUB.value})
    checkpoint = redact_secrets(task.checkpoint or {})
    checkpoint_json = json.dumps(checkpoint, ensure_ascii=False, separators=(",", ":"))
    if len(checkpoint_json) > 50_000:
        checkpoint_json = json.dumps({"version": 2, "steps": task.current_step, "repairs": task.repair_attempts, "truncated": True}, separators=(",", ":"))
    next_attempt = task.attempts + 1
    try:
        dispatch_result = dispatcher.dispatch(repository, workflow, ref, {"task_id": str(task.id), "task_prompt": task.prompt, "max_steps": str(task.max_steps), "repository": repository, "base_branch": str(task.metadata.get("base_branch", ref)), "working_branch": branch, "callback_url": callback_url, "checkpoint_json": checkpoint_json})
    except Exception:
        laptop = orchestrator.select_laptop() if settings.laptop_worker_enabled else None
        if laptop:
            task.metadata["worker_kind"] = WorkerKind.LAPTOP.value
            task.metadata["assigned_worker_id"] = laptop.worker_id
            task.status = TaskStatus.QUEUED
            task.error = "GitHub Actions dispatch failed; requeued for laptop failover"
            store.save(task)
            if hasattr(store, "event"):
                store.event(task.id, "github_dispatch_failed_laptop_fallback", {"worker_id": laptop.worker_id})
            return {"worker_kind": WorkerKind.LAPTOP.value, "worker_id": laptop.worker_id, "queued": True, "fallback": True}
        raise
    task.attempts = next_attempt
    task.status = TaskStatus.RUNNING
    task.error = None
    task.metadata["last_dispatch_idempotency_key"] = f"{task.id}:{next_attempt}"
    store.save(task)
    if hasattr(store, "event"):
        store.event(task.id, "worker_dispatched", {"repository": repository, "branch": branch, "attempt": task.attempts, "worker_kind": WorkerKind.GITHUB.value})
    return dispatch_result


def _dispatch(task: Task) -> dict:
    candidate = orchestrator.select() if settings.laptop_worker_enabled else None
    if candidate and candidate.kind == WorkerKind.LAPTOP:
        task.metadata["worker_kind"] = WorkerKind.LAPTOP.value
        task.metadata["assigned_worker_id"] = candidate.worker_id
        task.error = None
        task.status = TaskStatus.QUEUED
        store.save(task)
        if hasattr(store, "event"):
            store.event(task.id, "worker_assigned", {"worker_kind": candidate.kind.value, "worker_id": candidate.worker_id, "priority": candidate.priority})
        return {"worker_kind": candidate.kind.value, "worker_id": candidate.worker_id, "queued": True}
    if settings.github_token or settings.github_worker_repository:
        return _github_dispatch(task)
    task.metadata.pop("worker_kind", None)
    task.metadata.pop("assigned_worker_id", None)
    task.status = TaskStatus.QUEUED
    store.save(task)
    if hasattr(store, "event"):
        store.event(task.id, "worker_unavailable", {})
    return {"worker_kind": None, "queued": True, "reason": "no live laptop and GitHub Actions is not configured"}


@app.post("/api/tasks/{task_id}/dispatch")
def dispatch_task(task_id: UUID):
    task = store.get(task_id)
    if not task:
        raise HTTPException(404, "task not found")
    if task.status == TaskStatus.RUNNING:
        raise HTTPException(409, "task is already running")
    return {"task_id": str(task.id), **_dispatch(task)}


@app.post("/api/tasks/{task_id}/resume")
def resume_task(task_id: UUID):
    task = store.get(task_id)
    if not task:
        raise HTTPException(404, "task not found")
    if task.status == TaskStatus.RUNNING:
        raise HTTPException(409, "task is already running")
    if not task.checkpoint and task.status not in {TaskStatus.FAILED, TaskStatus.CHECKPOINTED}:
        raise HTTPException(409, "task has no resumable checkpoint")
    return {"task_id": str(task.id), "resumed": True, **_dispatch(task)}


@app.post("/api/tasks/{task_id}/cancel")
def cancel_task(task_id: UUID):
    task = store.get(task_id)
    if not task:
        raise HTTPException(404, "task not found")
    if task.status in {TaskStatus.COMPLETED, TaskStatus.CANCELLED}:
        return {"task_id": str(task.id), "cancelled": task.status == TaskStatus.CANCELLED, "status": task.status.value}
    remote = {"attempted": False, "cancel_requested": False}
    repository = str(task.metadata.get("repository") or settings.github_worker_repository or "")
    if task.worker_run_id and repository and settings.github_token:
        try:
            remote = {"attempted": True, **GitHubActionsDispatcher(settings.github_token).cancel_run(repository, task.worker_run_id)}
        except Exception as exc:
            remote = {"attempted": True, "cancel_requested": False, "error": redact_secrets(str(exc))}
    task.status = TaskStatus.CANCELLED
    task.error = "cancelled by user"
    task.metadata["cancelled_at"] = time.time()
    task.metadata["cancelled"] = True
    task.metadata["remote_cancel"] = remote
    lease_id = str(task.metadata.get("lease_id") or "")
    if lease_id:
        with suppress(Exception):
            leases.release_task(str(task.id)) if leases else local_leases.revoke(lease_id)
    store.save(task)
    if hasattr(store, "event"):
        store.event(task.id, "cancelled", {"by": "user", "remote_cancel": remote})
    return {"task_id": str(task.id), "cancelled": True, "status": task.status.value, "remote_cancel": remote}


@app.post("/api/tasks/{task_id}/run")
async def execute_task(task_id: UUID, workspace: str):
    task = store.get(task_id)
    if not task:
        raise HTTPException(404, "task not found")
    if task.status == TaskStatus.CANCELLED:
        raise HTTPException(409, "task is cancelled")
    task.status = TaskStatus.RUNNING
    store.save(task)
    try:
        result = await run_task(task, router, workspace)
    except Exception as exc:
        task.status = TaskStatus.FAILED
        task.error = redact_secrets(str(exc))
        store.save(task)
        raise HTTPException(500, task.error)
    task.status = TaskStatus.COMPLETED if result.get("status") == "completed" else TaskStatus.CHECKPOINTED if result.get("status") == "checkpointed" else TaskStatus.FAILED
    if result.get("error"):
        task.error = redact_secrets(str(result["error"]))
    store.save(task)
    if hasattr(store, "event"):
        store.event(task.id, task.status.value, {"step": task.current_step, "repairs": task.repair_attempts})
    return task


@app.post("/api/tasks/{task_id}/worker-callback")
async def worker_callback(task_id: UUID, request: Request, payload: dict, authorization: str | None = Header(default=None), x_agent_timestamp: str | None = Header(default=None), x_agent_signature: str | None = Header(default=None)):
    body = await request.body()
    if len(body) > settings.model_max_response_bytes:
        raise HTTPException(413, "callback payload too large")
    secret = settings.github_callback_token
    if secret:
        if not verify_callback_signature(secret, x_agent_timestamp or "", body, x_agent_signature or "", tolerance_seconds=settings.callback_signature_tolerance_seconds):
            raise HTTPException(401, "invalid or expired callback signature")
    else:
        _callback_authorized(authorization.removeprefix("Bearer ") if authorization else None)
    task = store.get(task_id)
    if not task:
        raise HTTPException(404, "task not found")
    run_id = str(payload.get("run_id") or "")
    accepted_runs = task.metadata.setdefault("accepted_worker_runs", [])
    if run_id and run_id in accepted_runs:
        return {"accepted": True, "duplicate": True, "status": task.status.value}
    if task.status == TaskStatus.CANCELLED:
        if hasattr(store, "event"):
            store.event(task.id, "late_worker_result_ignored", {"run_id": run_id})
        return {"accepted": True, "ignored": True, "status": task.status.value}
    expected_attempt = int(task.attempts)
    callback_attempt = int(payload.get("attempt", expected_attempt))
    if callback_attempt < expected_attempt:
        if hasattr(store, "event"):
            store.event(task.id, "stale_worker_result_ignored", {"run_id": run_id, "callback_attempt": callback_attempt, "expected_attempt": expected_attempt})
        return {"accepted": True, "ignored": True, "stale": True, "status": task.status.value}
    task.worker_id = payload.get("worker_id") or task.worker_id
    task.worker_run_id = run_id or task.worker_run_id
    task.current_step = int(payload.get("steps", task.current_step))
    task.repair_attempts = int(payload.get("repairs", task.repair_attempts))
    task.result = redact_secrets(payload)
    status = str(payload.get("status", "failed"))
    task.status = TaskStatus.COMPLETED if status == "completed" else TaskStatus.CHECKPOINTED if status == "checkpointed" else TaskStatus.FAILED
    task.error = redact_secrets(str(payload["error"])) if payload.get("error") else None
    if status == "checkpointed":
        task.checkpoint = {"version": 2, "task_id": str(task.id), "active_role": payload.get("active_role"), "phases": redact_secrets(payload.get("phases", [])), "messages": redact_secrets(payload.get("messages", [])), "steps": task.current_step, "repairs": task.repair_attempts, "run_id": run_id}
    lease_id = str(task.metadata.get("lease_id") or "")
    if lease_id:
        with suppress(Exception):
            leases.release_task(str(task.id)) if leases else local_leases.revoke(lease_id)
        task.metadata.pop("lease_id", None)
    if task.worker_id:
        with suppress(Exception):
            workers.release(task.worker_id)
    if run_id:
        accepted_runs.append(run_id)
        task.metadata["accepted_worker_runs"] = accepted_runs[-20:]
    store.save(task)
    if hasattr(store, "event"):
        store.event(task.id, "worker_result", {"status": status, "step": task.current_step, "attempt": task.attempts, "run_id": run_id})
    max_attempts = int(task.metadata.get("max_worker_attempts", settings.max_worker_attempts))
    if status == "checkpointed" and task.attempts < max_attempts and task.status != TaskStatus.CANCELLED:
        return {"accepted": True, "resume_scheduled": True, **_dispatch(task)}
    return {"accepted": True, "resume_scheduled": False, "status": task.status.value}


@app.get("/api/tasks/{task_id}/stream")
async def stream_task(task_id: UUID):
    async def events():
        last = ""
        for _ in range(3600):
            task = store.get(task_id)
            if not task:
                yield "event: error\ndata: task not found\n\n"
                return
            state = task.model_dump_json()
            if state != last:
                yield f"event: task\ndata: {state}\n\n"
                last = state
            if task.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}:
                return
            await asyncio.sleep(1)
    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/tasks/{task_id}/checkpoint")
def checkpoint_task(task_id: UUID, step: int = 0):
    try:
        return store.checkpoint(task_id, step, {"version": 2, "resumable": True})
    except KeyError:
        raise HTTPException(404, "task not found")


@app.post("/api/workers/register")
def register_worker(worker: Worker, authorization: str | None = Header(default=None)):
    _authorized(authorization.removeprefix("Bearer ") if authorization else None)
    if not settings.laptop_worker_enabled:
        raise HTTPException(503, "laptop workers are disabled")
    return workers.register(worker)


@app.post("/api/workers/{worker_id}/heartbeat")
def heartbeat(worker_id: str, status: WorkerStatus | None = None, lease_id: str | None = None, authorization: str | None = Header(default=None)):
    _authorized(authorization.removeprefix("Bearer ") if authorization else None)
    try:
        result = workers.heartbeat(worker_id, status)
    except KeyError:
        raise HTTPException(404, "worker not found")
    if lease_id:
        renewed = leases.renew(worker_id, lease_id, settings.worker_lease_seconds) if leases else local_leases.renew(lease_id, settings.worker_lease_seconds)
        result["lease_renewed"] = bool(renewed)
        if renewed:
            now = time.time()
            for task in store.list():
                if task.worker_id == worker_id and task.status == TaskStatus.RUNNING and str(task.metadata.get("lease_id") or "") == lease_id:
                    task.metadata["worker_heartbeat_at"] = now
                    store.save(task)
    return result


def _claim_lease(worker_id: str, task_id: UUID) -> str:
    lease_id = uuid4().hex
    if leases:
        if not leases.claim(worker_id, str(task_id), lease_id, settings.worker_lease_seconds):
            raise HTTPException(409, "task already leased")
    else:
        local_leases.issue(worker_id, str(task_id), lease_id, settings.worker_lease_seconds)
    return lease_id


@app.post("/api/tasks/claim-next")
def claim_next_task(authorization: str | None = Header(default=None)):
    _authorized(authorization.removeprefix("Bearer ") if authorization else None)
    if not settings.laptop_worker_enabled:
        raise HTTPException(503, "laptop workers are disabled")
    candidates = orchestrator.candidates()
    laptop_ids = {c.worker_id for c in candidates if c.kind == WorkerKind.LAPTOP}
    if not laptop_ids:
        return Response(status_code=204)
    for task in store.list():
        if task.status not in {TaskStatus.QUEUED, TaskStatus.CHECKPOINTED}:
            continue
        kind = str(task.metadata.get("worker_kind") or "")
        assigned = str(task.metadata.get("assigned_worker_id") or "")
        if kind not in {"", WorkerKind.LAPTOP.value}:
            continue
        worker_id = assigned if assigned in laptop_ids else sorted(laptop_ids)[0]
        lease_id = _claim_lease(worker_id, task.id)
        task.status = TaskStatus.RUNNING
        task.attempts += 1
        task.worker_id = worker_id
        task.worker_run_id = None
        task.error = None
        task.metadata["worker_kind"] = WorkerKind.LAPTOP.value
        task.metadata["assigned_worker_id"] = worker_id
        task.metadata["lease_id"] = lease_id
        task.metadata["worker_heartbeat_at"] = time.time()
        store.save(task)
        if hasattr(store, "event"):
            store.event(task.id, "worker_claimed", {"worker_id": worker_id, "attempt": task.attempts})
        return {"task": task.model_dump(mode="json"), "lease_id": lease_id, "attempt": task.attempts}
    return Response(status_code=204)


@app.post("/api/workers/{worker_id}/claim/{task_id}")
def claim_task(worker_id: str, task_id: UUID, authorization: str | None = Header(default=None)):
    _authorized(authorization.removeprefix("Bearer ") if authorization else None)
    if not settings.laptop_worker_enabled:
        raise HTTPException(503, "laptop workers are disabled")
    if worker_id not in {c.worker_id for c in orchestrator.candidates() if c.kind == WorkerKind.LAPTOP}:
        raise HTTPException(409, "worker is not online")
    task = store.get(task_id)
    if not task:
        raise HTTPException(404, "task not found")
    if task.status not in {TaskStatus.QUEUED, TaskStatus.CHECKPOINTED}:
        raise HTTPException(409, "task is not claimable")
    lease_id = _claim_lease(worker_id, task_id)
    task.status = TaskStatus.RUNNING
    task.attempts += 1
    task.worker_id = worker_id
    task.metadata["worker_kind"] = WorkerKind.LAPTOP.value
    task.metadata["assigned_worker_id"] = worker_id
    task.metadata["lease_id"] = lease_id
    task.metadata["worker_heartbeat_at"] = time.time()
    task.error = None
    store.save(task)
    return {"task": task.model_dump(mode="json"), "worker_id": worker_id, "lease_id": lease_id, "attempt": task.attempts, "ttl_seconds": settings.worker_lease_seconds}


@app.delete("/api/workers/{worker_id}/lease/{lease_id}")
def release_lease(worker_id: str, lease_id: str, authorization: str | None = Header(default=None)):
    _authorized(authorization.removeprefix("Bearer ") if authorization else None)
    released = leases.release(worker_id, lease_id) if leases else local_leases.revoke(lease_id)
    return {"released": bool(released)}


@app.post("/api/workers/{worker_id}/lease/{lease_id}/renew")
def renew_lease(worker_id: str, lease_id: str, authorization: str | None = Header(default=None)):
    _authorized(authorization.removeprefix("Bearer ") if authorization else None)
    renewed = leases.renew(worker_id, lease_id, settings.worker_lease_seconds) if leases else local_leases.renew(lease_id, worker_id, settings.worker_lease_seconds)
    return {"renewed": bool(renewed)}


@app.post("/api/workers/recover")
def recover_workers(authorization: str | None = Header(default=None)):
    _authorized(authorization.removeprefix("Bearer ") if authorization else None)
    recovered = _recover_once()
    return {"recovered": recovered, "count": len(recovered)}
