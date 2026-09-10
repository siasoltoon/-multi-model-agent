from __future__ import annotations

import asyncio
import os
import httpx
from uuid import uuid4
from .discovery import ProviderDiscovery
from .models import Task
from .router import ModelEndpoint, SmartRouter
from .runner import run_task


async def _callback(result: dict) -> None:
    base = os.getenv("AGENT_CALLBACK_URL", "").rstrip("/")
    task_id = os.getenv("AGENT_TASK_ID", "")
    if not base or not task_id:
        return
    url = f"{base}/api/tasks/{task_id}/worker-callback"
    headers = {}
    token = os.getenv("AGENT_CALLBACK_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = dict(result)
    payload["task_id"] = task_id
    payload["worker_id"] = os.getenv("AGENT_WORKER_ID", "github-actions")
    payload["run_id"] = os.getenv("GITHUB_RUN_ID", "")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
    except Exception as exc:
        print(f"control-plane callback warning: {exc}")


async def main() -> int:
    workspace = os.getenv("AGENT_WORKSPACE", os.getcwd())
    prompt = os.getenv("AGENT_TASK_PROMPT", "")
    if not prompt:
        raise SystemExit("AGENT_TASK_PROMPT is required")
    router = SmartRouter()
    for item in await ProviderDiscovery().discover():
        router.register(ModelEndpoint(
            id=f"{item.provider}:{item.model}:{item.base_url}", provider=item.provider,
            model=item.model, base_url=item.base_url, context_window=item.context_window, tool_support=item.tool_support,
            task_fit=item.task_fit, reliability=item.reliability, latency_ms=item.latency_ms,
            api_key_env=item.api_key_env, metadata=item.metadata,
        ))
    if not router.endpoints:
        base, model = os.getenv("AGENT_BASE_URL", ""), os.getenv("AGENT_MODEL", "")
        if base and model:
            router.register(ModelEndpoint("env", "env", model, base_url=base, tool_support=True))
    if not router.endpoints:
        raise SystemExit("No model endpoint discovered")
    task = Task(id=uuid4(), prompt=prompt, max_steps=int(os.getenv("AGENT_MAX_STEPS", "64")))
    task_id = os.getenv("AGENT_TASK_ID", "")
    if task_id:
        from uuid import UUID
        task.id = UUID(task_id)
    result = await run_task(task, router, workspace)
    print(result)
    await _callback(result)
    return 0 if result.get("status") in {"completed", "checkpointed"} else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
