from __future__ import annotations

import os
from time import monotonic
from typing import Any

from .adapters import OpenAICompatibleAdapter
from .agent_loop import AgentLoop, AgentPolicy
from .discovery import env_api_key
from .models import Task
from .router import SmartRouter
from .workspace_tools import WorkspaceTools


def _task_type(prompt: str, metadata: dict[str, Any]) -> str:
    explicit = metadata.get("task_type")
    if explicit:
        return str(explicit)
    text = prompt.lower()
    if any(x in text for x in ("review", "code review", "بررسی کد", "بازبینی")):
        return "review"
    if any(x in text for x in ("test", "testing", "تست", "آزمون")):
        return "testing"
    if any(x in text for x in ("document", "documentation", "readme", "مستند", "راهنما")):
        return "documentation"
    return "coding"


async def run_task(task: Task, router: SmartRouter, workspace: str) -> dict[str, Any]:
    task_type = _task_type(task.prompt, task.metadata)
    endpoint = router.choose(min_context=4096, tools=True, task_type=task_type)
    api_key = env_api_key(getattr(endpoint, "api_key_env", None)) or os.getenv(f"{endpoint.provider.upper()}_API_KEY", "") or os.getenv("AGENT_API_KEY", "")
    adapter = OpenAICompatibleAdapter(endpoint.base_url, api_key, endpoint.model, timeout=float(os.getenv("AGENT_MODEL_REQUEST_TIMEOUT", "180")))
    tools = WorkspaceTools(workspace, command_timeout=float(os.getenv("AGENT_COMMAND_TIMEOUT", "300")))
    system = ("You are a senior software engineer. Work directly in the provided workspace. "
              "Inspect before editing, make minimal correct changes, run relevant tests, inspect git diff, "
              "and keep repairing failures until the task is genuinely complete. Never claim success without verification.")
    checkpoint_messages = task.checkpoint.get("messages") if isinstance(task.checkpoint, dict) else None
    messages = checkpoint_messages if isinstance(checkpoint_messages, list) and checkpoint_messages else [{"role": "system", "content": system}, {"role": "user", "content": task.prompt}]
    timeout = float(os.getenv("AGENT_TIMEOUT_SECONDS", "1800"))
    loop = AgentLoop(adapter, tools.as_tools(), AgentPolicy(max_steps=task.max_steps, repair_attempts=6, timeout_seconds=timeout))
    started = monotonic()
    try:
        result = await loop.run(messages, tools.specs())
    except Exception as exc:
        router.mark_failure(endpoint.id, exc)
        raise
    elapsed_ms = (monotonic() - started) * 1000.0
    if result.get("status") == "completed":
        router.mark_success(endpoint.id, latency_ms=elapsed_ms)
    elif result.get("status") == "failed":
        router.mark_failure(endpoint.id, result.get("error", "agent failed"))
    task.current_step = result.get("steps", task.current_step)
    task.repair_attempts = result.get("repairs", task.repair_attempts)
    task.result = result
    task.metadata["last_model"] = endpoint.model
    task.metadata["last_provider"] = endpoint.provider
    task.metadata["task_type"] = task_type
    task.metadata["last_latency_ms"] = round(elapsed_ms, 2)
    if result.get("status") == "checkpointed":
        task.checkpoint = {"messages": result.get("messages", messages), "steps": task.current_step, "repairs": task.repair_attempts}
    return result
