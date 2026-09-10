import uuid
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.core.models import Task, TaskStatus
from app.core.planner import DAGPlanner
from app.core.router import SmartRouter

app = FastAPI(title="Multi-Model Agent", version="0.1.0")
planner = DAGPlanner()
router = SmartRouter()
tasks: dict[str, Task] = {}


class TaskRequest(BaseModel):
    prompt: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "multi-model-agent"}


@app.post("/api/tasks")
def create_task(request: TaskRequest) -> dict:
    task_id = uuid.uuid4().hex
    task = Task(id=task_id, prompt=request.prompt, status=TaskStatus.PLANNING)
    tasks[task_id] = task
    nodes = planner.plan(request.prompt)
    task.status = TaskStatus.QUEUED
    return {"task_id": task_id, "status": task.status, "plan": [n.id for n in nodes]}


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str) -> dict:
    task = tasks[task_id]
    return {"id": task.id, "prompt": task.prompt, "status": task.status, "checkpoint": task.checkpoint}


@app.get("/terminal", response_class=HTMLResponse)
def terminal() -> str:
    return '''<!doctype html><html><head><meta charset="utf-8"><title>Agent Terminal</title>
<style>body{font-family:monospace;background:#111;color:#eee;margin:0;padding:24px}textarea{width:100%;height:120px;background:#181818;color:#eee;border:1px solid #444;padding:12px}button{margin-top:10px;padding:10px 18px}pre{white-space:pre-wrap}</style></head>
<body><h1>Multi-Model Agent</h1><p>Web Terminal</p><textarea id="p" placeholder="Describe the coding task..."></textarea><br><button onclick="run()">Run task</button><pre id="o"></pre>
<script>async function run(){const p=document.getElementById('p').value;const r=await fetch('/api/tasks',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:p})});document.getElementById('o').textContent=JSON.stringify(await r.json(),null,2)}</script></body></html>'''
