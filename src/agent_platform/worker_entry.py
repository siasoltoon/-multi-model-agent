from __future__ import annotations

import asyncio
import json
import os
from uuid import uuid4
from .discovery import ProviderDiscovery
from .models import Task
from .router import ModelEndpoint, SmartRouter
from .runner import run_task


async def main() -> int:
    workspace = os.getenv("AGENT_WORKSPACE", os.getcwd())
    prompt = os.getenv("AGENT_TASK_PROMPT", "")
    if not prompt:
        raise SystemExit("AGENT_TASK_PROMPT is required")
    router = SmartRouter()
    for item in await ProviderDiscovery().discover():
        router.register(ModelEndpoint(
            id=f"{item.provider}:{item.model}:{item.base_url}", provider=item.provider, model=item.model,
            base_url=item.base_url, context_window=item.context_window, tool_support=item.tool_support,
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
    checkpoint = os.getenv("AGENT_CHECKPOINT_JSON", "")
    if checkpoint:
        try:
            decoded = json.loads(checkpoint)
            if isinstance(decoded, dict):
                task.checkpoint = decoded
                task.current_step = int(decoded.get("steps", 0))
                task.repair_attempts = int(decoded.get("repairs", 0))
        except (json.JSONDecodeError, TypeError, ValueError):
            raise SystemExit("AGENT_CHECKPOINT_JSON is invalid")
    result = await run_task(task, router, workspace)
    result_path = os.path.join(workspace, ".agent-result.json")
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump({
            **result,
            "task_id": str(task.id),
            "worker_id": os.getenv("AGENT_WORKER_ID", "github-actions"),
            "run_id": os.getenv("GITHUB_RUN_ID", ""),
        }, handle, ensure_ascii=False, default=str)
    print(result)
    return 0 if result.get("status") in {"completed", "checkpointed"} else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
