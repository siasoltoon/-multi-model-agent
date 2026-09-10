from __future__ import annotations

import os
from typing import Any

from .adapters import OpenAICompatibleAdapter
from .agent_loop import AgentLoop, AgentPolicy
from .discovery import env_api_key
from .models import Task
from .router import SmartRouter
from .workspace_tools import WorkspaceTools


async def run_task(task: Task, router: SmartRouter, workspace: str) -> dict[str, Any]:
    endpoint = router.choose(min_context=4096, tools=True)
    api_key = (
        env_api_key(getattr(endpoint, "api_key_env", None))
        or os.getenv(f"{endpoint.provider.upper()}_API_KEY", "")
        or os.getenv("AGENT_API_KEY", "")
    )
    adapter = OpenAICompatibleAdapter(endpoint.base_url, api_key, endpoint.model)
    tools = WorkspaceTools(workspace)
    system = (
        "You are a senior software engineer. Work directly in the provided workspace. "
        "Inspect before editing, make minimal correct changes, run relevant tests, inspect git diff, "
        "and keep repairing failures until the task is genuinely complete. Never claim success without verification."
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": task.prompt}]
    loop = AgentLoop(adapter, tools.as_tools(), AgentPolicy(max_steps=task.max_steps, repair_attempts=6, timeout_seconds=1800))
    result = await loop.run(messages, tools.specs())
    router.update_health(endpoint.id, "ONLINE")
    task.current_step = result.get("steps", task.current_step)
    task.repair_attempts = result.get("repairs", task.repair_attempts)
    task.result = result
    return result
