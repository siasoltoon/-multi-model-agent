from __future__ import annotations

import asyncio
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
    result = await run_task(task, router, workspace)
    print(result)
    return 0 if result.get("status") in {"completed", "checkpointed"} else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
