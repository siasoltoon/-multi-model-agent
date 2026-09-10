from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx

from .discovery import ProviderDiscovery
from .models import Task
from .router import ModelEndpoint, SmartRouter
from .runner import run_task


def _headers() -> dict[str, str]:
    token = os.getenv("AGENT_WORKER_AUTH_TOKEN", "")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _base_url() -> str:
    return os.getenv("AGENT_CONTROL_PLANE_URL", "http://127.0.0.1:8000").rstrip("/")


def _print(value: object) -> None:
    if isinstance(value, str):
        print(value)
    else:
        print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def submit(prompt: str, steps: int | None, repository: str | None, base_branch: str | None) -> int:
    metadata = {}
    if repository:
        metadata["repository"] = repository
    if base_branch:
        metadata["base_branch"] = base_branch
    payload = {"prompt": prompt, "metadata": metadata}
    if steps:
        payload["max_steps"] = steps
    with httpx.Client(timeout=30) as client:
        response = client.post(f"{_base_url()}/api/tasks", json=payload, headers=_headers())
        response.raise_for_status()
        task = response.json()
        response = client.post(f"{_base_url()}/api/tasks/{task['id']}/dispatch", headers=_headers())
        response.raise_for_status()
        _print({"task_id": task["id"], "dispatched": response.json()})
    return 0


def status(task_id: str) -> int:
    with httpx.Client(timeout=30) as client:
        response = client.get(f"{_base_url()}/api/tasks/{task_id}", headers=_headers())
        response.raise_for_status()
        _print(response.json())
    return 0


def watch(task_id: str, interval: float) -> int:
    last = None
    with httpx.Client(timeout=30) as client:
        while True:
            response = client.get(f"{_base_url()}/api/tasks/{task_id}", headers=_headers())
            response.raise_for_status()
            task = response.json()
            state = (task.get("status"), task.get("current_step"), task.get("repair_attempts"), task.get("error"))
            if state != last:
                print(f"[{state[0]}] step={state[1]} repairs={state[2]}")
                if task.get("error"):
                    print(f"error: {task['error']}")
                last = state
            if state[0] in {"completed", "failed", "cancelled"}:
                return 0 if state[0] == "completed" else 2
            time.sleep(interval)


async def run_local(prompt: str, workspace: str, steps: int) -> int:
    router = SmartRouter()
    for item in await ProviderDiscovery().discover():
        router.register(ModelEndpoint(
            id=f"{item.provider}:{item.model}:{item.base_url}",
            provider=item.provider,
            model=item.model,
            base_url=item.base_url,
            context_window=item.context_window,
            tool_support=item.tool_support,
            task_fit=item.task_fit,
            reliability=item.reliability,
            latency_ms=item.latency_ms,
            api_key_env=item.api_key_env,
            metadata=item.metadata,
        ))
    if not router.endpoints:
        base = os.getenv("AGENT_BASE_URL", "")
        model = os.getenv("AGENT_MODEL", "")
        if base and model:
            router.register(ModelEndpoint("env", "env", model, base_url=base, tool_support=True))
    if not router.endpoints:
        print("No model endpoint discovered. Configure Ollama or AGENT_BASE_URL/AGENT_MODEL.", file=sys.stderr)
        return 2
    task = Task(prompt=prompt, max_steps=steps)
    print(f"task={task.id} workspace={Path(workspace).resolve()}")
    result = await run_task(task, router, workspace)
    _print(result)
    return 0 if result.get("status") == "completed" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="multi-model-agent", description="Terminal-first multi-model coding agent")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("submit", help="create a task and dispatch a GitHub Actions worker")
    p.add_argument("prompt")
    p.add_argument("--steps", type=int, choices=(32, 64), default=None)
    p.add_argument("--repository")
    p.add_argument("--base-branch", default="main")

    p = sub.add_parser("status", help="show task state")
    p.add_argument("task_id")

    p = sub.add_parser("watch", help="watch a remote task until completion")
    p.add_argument("task_id")
    p.add_argument("--interval", type=float, default=2.0)

    p = sub.add_parser("run-local", help="run the agent directly in this terminal/workspace")
    p.add_argument("prompt")
    p.add_argument("--workspace", default=os.getenv("AGENT_WORKSPACE", os.getcwd()))
    p.add_argument("--steps", type=int, choices=(32, 64), default=64)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "submit":
        return submit(args.prompt, args.steps, args.repository, args.base_branch)
    if args.command == "status":
        return status(args.task_id)
    if args.command == "watch":
        return watch(args.task_id, args.interval)
    if args.command == "run-local":
        return asyncio.run(run_local(args.prompt, args.workspace, args.steps))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
