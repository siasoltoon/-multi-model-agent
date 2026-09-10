from __future__ import annotations

import asyncio
from uuid import UUID, uuid4
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import JSONResponse, StreamingResponse
from .config import settings
from .discovery import ProviderDiscovery
from .engine import AgentEngine
from .github_dispatcher import GitHubActionsDispatcher
from .models import HealthResponse, Task, TaskRequest, TaskStatus
from .persistent_store import PersistentTaskStore
from .router import ModelEndpoint, SmartRouter
from .runner import run_task
from .workers import Worker, WorkerRegistry, WorkerStatus


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
leases = None
if settings.database_url.startswith(("postgresql://", "postgres://")):
    from .persistent_leases import PersistentWorkerLeases
    leases = PersistentWorkerLeases(settings.database_url)
app = FastAPI(title=settings.app_name, version="0.5.0")


def _authorized(token: str | None) -> None:
    if settings.worker_auth_token and token != settings.worker_auth_token:
        raise HTTPException(401, "invalid worker token")


def _callback_authorized(token: str | None) -> None:
    expected = settings.github_callback_token or settings.worker_auth_token
    if expected and token != expected:
        raise HTTPException(401, "invalid callback token")


def _register(items):
    for item in items:
        router.register(ModelEndpoint(
            id=f"{item.provider}:{item.model}:{item.base_url}", provider=item.provider,
            model=item.model, base_url=item.base_url, context_window=item.context_window,
            tool_support=item.tool_support, task_fit=item.task_fit, reliability=item.reliability,
            latency_ms=item.latency_ms, quota_remaining=1.0, api_key_env=item.api_key_env, metadata=item.metadata,
        ))


@app.on_event("startup")
async def startup_discovery():
    _register(await discovery.discover())


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
    return JSONResponse({
        "name": settings.app_name,
        "interface": "terminal",
        "message": "Web UI is intentionally disabled. Use the multi-model-agent CLI.",
        "commands": ["multi-model-agent submit", "multi-model-agent watch", "multi-model-agent status", "multi-model-agent run-local"],
    })


@app.post("/api/providers/discover")
async def discover_providers():
    items = await discovery.discover(); _register(items)
    return {"count": len(items), "providers": sorted({x.provider for x in items}), "models": sorted({x.model for x in items})}


@app.get("/api/providers")
def providers():
    return [{"id": e.id, "provider": e.provider, "model": e.model, "base_url": e.base_url, "health": e.health, "context_window": e.context_window, "tool_support": e.tool_support} for e in router.endpoints]


@app.post("/api/tasks", response_model=Task)
def create_task(request: TaskRequest):
    task = Task(prompt=request.prompt, max_steps=request.max_steps or settings.max_agent_steps, metadata=request.metadata)
    store.save(task)
    if hasattr(store, "event"): store.event(task.id, "created", {"max_steps": task.max_steps})
    if settings.redis_url:
        try:
            from .queue import RedisTaskQueue
            RedisTaskQueue(settings.redis_url).enqueue(str(task.id))
        except Exception: pass
    return task


@app.get("/api/tasks", response_model=list[Task])
def list_tasks(): return store.list()


@app.get("/api/tasks/{task_id}", response_model=Task)
def get_task(task_id: UUID):
    task = store.get(task_id)
    if not task: raise HTTPException(404, "task not found")
    return task


@app.get("/api/tasks/{task_id}/events")
def task_events(task_id: UUID):
    if not hasattr(store, "events"): raise HTTPException(501, "event listing unavailable")
    return store.events(task_id)


@app.post("/api/tasks/{task_id}/plan")
def plan_task(task_id: UUID):
    task = store.get(task_id)
    if not task: raise HTTPException(404, "task not found")
    plan = engine.plan(task); task.status = TaskStatus.PLANNING; store.save(task)
    return {"task_id": str(task.id), "nodes": [{"id": n.id, "role": n.role, "dependencies": sorted(n.dependencies)} for n in plan.dag.nodes.values()]}


def _dispatch(task: Task) -> dict:
    repository = str(task.metadata.get("repository") or settings.github_worker_repository)
    if not repository:
        raise HTTPException(400, "repository is required in task metadata or AGENT_GITHUB_WORKER_REPOSITORY")
    dispatcher = GitHubActionsDispatcher(settings.github_token)
    branch = str(task.metadata.get("worker_branch") or f"agent/task-{task.id}")
    callback_url = str(task.metadata.get("callback_url") or settings.public_base_url).rstrip("/")
    if not callback_url:
        raise HTTPException(400, "AGENT_PUBLIC_BASE_URL is required for automatic worker callbacks")
    task.metadata.update({"repository": repository, "worker_branch": branch, "worker_workflow": settings.github_worker_workflow, "callback_url": callback_url})
    task.attempts += 1
    task.status = TaskStatus.RUNNING
    store.save(task)
    if hasattr(store, "event"): store.event(task.id, "worker_dispatched", {"repository": repository, "branch": branch, "attempt": task.attempts})
    return dispatcher.dispatch(repository, settings.github_worker_workflow, settings.github_worker_ref, {
        "task_id": str(task.id), "task_prompt": task.prompt, "max_steps": str(task.max_steps),
        "repository": repository, "base_branch": str(task.metadata.get("base_branch", settings.github_worker_ref)),
        "working_branch": branch, "callback_url": callback_url,
    })


@app.post("/api/tasks/{task_id}/dispatch")
def dispatch_task(task_id: UUID):
    task = store.get(task_id)
    if not task: raise HTTPException(404, "task not found")
    return {"task_id": str(task.id), **_dispatch(task)}


@app.post("/api/tasks/{task_id}/run")
async def execute_task(task_id: UUID, workspace: str):
    task = store.get(task_id)
    if not task: raise HTTPException(404, "task not found")
    task.status = TaskStatus.RUNNING; store.save(task)
    try: result = await run_task(task, router, workspace)
    except Exception as exc:
        task.status = TaskStatus.FAILED; task.error = str(exc); store.save(task); raise HTTPException(500, str(exc))
    task.status = TaskStatus.COMPLETED if result.get("status") == "completed" else TaskStatus.CHECKPOINTED if result.get("status") == "checkpointed" else TaskStatus.FAILED
    if result.get("error"): task.error = str(result["error"])
    store.save(task)
    if hasattr(store, "event"): store.event(task.id, task.status.value, {"step": task.current_step, "repairs": task.repair_attempts})
    return task


@app.post("/api/tasks/{task_id}/worker-callback")
def worker_callback(task_id: UUID, payload: dict, authorization: str | None = Header(default=None)):
    _callback_authorized(authorization.removeprefix("Bearer ") if authorization else None)
    task = store.get(task_id)
    if not task: raise HTTPException(404, "task not found")
    task.worker_id = payload.get("worker_id") or task.worker_id
    task.worker_run_id = payload.get("run_id") or task.worker_run_id
    task.current_step = int(payload.get("steps", task.current_step))
    task.repair_attempts = int(payload.get("repairs", task.repair_attempts))
    task.result = payload
    status = str(payload.get("status", "failed"))
    task.status = TaskStatus.COMPLETED if status == "completed" else TaskStatus.CHECKPOINTED if status == "checkpointed" else TaskStatus.FAILED
    task.error = str(payload["error"]) if payload.get("error") else None
    if status == "checkpointed":
        task.checkpoint = {"messages": payload.get("messages", []), "steps": task.current_step, "repairs": task.repair_attempts}
    store.save(task)
    if hasattr(store, "event"): store.event(task.id, "worker_result", {"status": status, "step": task.current_step, "attempt": task.attempts})
    max_attempts = int(task.metadata.get("max_worker_attempts", 5))
    if status == "checkpointed" and task.attempts < max_attempts:
        return {"accepted": True, "resume_scheduled": True, **_dispatch(task)}
    return {"accepted": True, "resume_scheduled": False, "status": task.status.value}


@app.get("/api/tasks/{task_id}/stream")
async def stream_task(task_id: UUID):
    async def events():
        last = ""
        for _ in range(3600):
            task = store.get(task_id)
            if not task: yield "event: error\ndata: task not found\n\n"; return
            state = task.model_dump_json()
            if state != last: yield f"event: task\ndata: {state}\n\n"; last = state
            if task.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}: return
            await asyncio.sleep(1)
    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control":"no-cache", "X-Accel-Buffering":"no"})


@app.post("/api/tasks/{task_id}/checkpoint")
def checkpoint_task(task_id: UUID, step: int = 0):
    try: return store.checkpoint(task_id, step, {"resumable": True})
    except KeyError: raise HTTPException(404, "task not found")


@app.post("/api/workers/register")
def register_worker(worker: Worker, authorization: str | None = Header(default=None)):
    _authorized(authorization.removeprefix("Bearer ") if authorization else None); return workers.register(worker)


@app.post("/api/workers/{worker_id}/heartbeat")
def heartbeat(worker_id: str, status: WorkerStatus | None = None, authorization: str | None = Header(default=None)):
    _authorized(authorization.removeprefix("Bearer ") if authorization else None)
    try: return workers.heartbeat(worker_id, status)
    except KeyError: raise HTTPException(404, "worker not found")


@app.post("/api/workers/{worker_id}/claim/{task_id}")
def claim_task(worker_id: str, task_id: UUID, authorization: str | None = Header(default=None)):
    _authorized(authorization.removeprefix("Bearer ") if authorization else None)
    if not leases: raise HTTPException(501, "persistent leases require PostgreSQL")
    lease_id = uuid4().hex
    if not leases.claim(worker_id, str(task_id), lease_id, settings.worker_lease_seconds): raise HTTPException(409, "task already leased")
    return {"task_id": str(task_id), "worker_id": worker_id, "lease_id": lease_id, "ttl_seconds": settings.worker_lease_seconds}


@app.post("/api/workers/{worker_id}/lease/{lease_id}/renew")
def renew_lease(worker_id: str, lease_id: str, authorization: str | None = Header(default=None)):
    _authorized(authorization.removeprefix("Bearer ") if authorization else None)
    if not leases: raise HTTPException(501, "persistent leases require PostgreSQL")
    return {"renewed": leases.renew(worker_id, lease_id, settings.worker_lease_seconds)}


@app.delete("/api/workers/{worker_id}/lease/{lease_id}")
def release_lease(worker_id: str, lease_id: str, authorization: str | None = Header(default=None)):
    _authorized(authorization.removeprefix("Bearer ") if authorization else None)
    if not leases: raise HTTPException(501, "persistent leases require PostgreSQL")
    return {"released": leases.release(worker_id, lease_id)}
