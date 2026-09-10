from __future__ import annotations
import asyncio
from uuid import UUID
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from .config import settings
from .discovery import ProviderDiscovery, env_api_key
from .engine import AgentEngine
from .models import HealthResponse, Task, TaskRequest, TaskStatus
from .persistent_store import PersistentTaskStore
from .router import ModelEndpoint, SmartRouter
from .workers import Worker, WorkerRegistry, WorkerStatus

store = PersistentTaskStore(settings.database_url)
workers = WorkerRegistry()
router = SmartRouter()
engine = AgentEngine(router)
discovery = ProviderDiscovery()
app = FastAPI(title=settings.app_name, version="0.2.0")

@app.on_event("startup")
async def startup_discovery():
    for item in await discovery.discover():
        router.register(ModelEndpoint(
            id=f"{item.provider}:{item.model}:{item.base_url}", provider=item.provider,
            model=item.model, base_url=item.base_url, context_window=item.context_window,
            tool_support=item.tool_support, task_fit=item.task_fit, reliability=item.reliability,
            latency_ms=item.latency_ms, quota_remaining=1.0,
        ))

@app.get("/health", response_model=HealthResponse)
def health(): return HealthResponse()

@app.get("/", response_class=HTMLResponse)
def terminal():
    return """<!doctype html><meta charset='utf-8'><title>Multi-Model Agent</title><style>body{font-family:system-ui;margin:2rem;max-width:1100px}textarea{width:100%;height:180px}button{padding:.7rem 1rem;margin:.5rem 0}pre{white-space:pre-wrap;background:#f4f4f4;padding:1rem}</style><h1>Multi-Model Agent</h1><p>Web Terminal · automatic provider/model discovery</p><textarea id='p' placeholder='Describe the coding task...'></textarea><br><button onclick='go()'>Create task</button> <button onclick='discover()'>Refresh APIs</button><pre id='o'></pre><script>async function go(){let r=await fetch('/api/tasks',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:p.value})});o.textContent=JSON.stringify(await r.json(),null,2)}async function discover(){let r=await fetch('/api/providers/discover',{method:'POST'});o.textContent=JSON.stringify(await r.json(),null,2)}</script>"""

@app.post("/api/providers/discover")
async def discover_providers():
    items = await discovery.discover()
    for item in items:
        router.register(ModelEndpoint(id=f"{item.provider}:{item.model}:{item.base_url}", provider=item.provider, model=item.model, base_url=item.base_url, context_window=item.context_window, tool_support=item.tool_support, task_fit=item.task_fit, reliability=item.reliability, latency_ms=item.latency_ms))
    return {"count": len(items), "providers": [x.provider for x in items], "models": [x.model for x in items]}

@app.get("/api/providers")
def providers():
    return [{"id": e.id, "provider": e.provider, "model": e.model, "base_url": e.base_url, "health": e.health, "context_window": e.context_window, "tool_support": e.tool_support} for e in router.endpoints]

@app.post("/api/tasks", response_model=Task)
def create_task(request: TaskRequest):
    task = Task(prompt=request.prompt, max_steps=request.max_steps or settings.max_agent_steps, metadata=request.metadata)
    return store.save(task)

@app.get("/api/tasks", response_model=list[Task])
def list_tasks(): return store.list()

@app.get("/api/tasks/{task_id}", response_model=Task)
def get_task(task_id: UUID):
    task = store.get(task_id)
    if not task: raise HTTPException(404, "task not found")
    return task

@app.post("/api/tasks/{task_id}/plan")
def plan_task(task_id: UUID):
    task = store.get(task_id)
    if not task: raise HTTPException(404, "task not found")
    plan = engine.plan(task); task.status = TaskStatus.PLANNING; store.save(task)
    return {"task_id": str(task.id), "nodes": [{"id": n.id, "role": n.role, "dependencies": sorted(n.dependencies)} for n in plan.dag.nodes.values()]}

@app.post("/api/tasks/{task_id}/checkpoint")
def checkpoint_task(task_id: UUID, step: int = 0):
    try: return store.checkpoint(task_id, step, {"resumable": True})
    except KeyError: raise HTTPException(404, "task not found")

@app.post("/api/workers/register")
def register_worker(worker: Worker): return workers.register(worker)

@app.post("/api/workers/{worker_id}/heartbeat")
def heartbeat(worker_id: str, status: WorkerStatus | None = None):
    try: return workers.heartbeat(worker_id, status)
    except KeyError: raise HTTPException(404, "worker not found")
