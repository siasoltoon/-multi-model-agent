from __future__ import annotations

from .models import Task
from .phase_runner import PhaseRunner
from .reliability import redact_secrets
from .router import SmartRouter


async def run_task(task: Task, router: SmartRouter, workspace: str) -> dict:
    """Run a task through the automatic role-based multi-model pipeline."""
    result = await PhaseRunner(router, workspace).run(task)
    task.current_step = int(result.get("steps", task.current_step) or 0)
    task.repair_attempts = int(result.get("repairs", task.repair_attempts) or 0)
    task.result = redact_secrets(result)
    phases = result.get("phases", [])
    task.metadata["execution_mode"] = "role_based_multi_model"
    task.metadata["roles_completed"] = [p.get("role") for p in phases if p.get("status") == "completed"]
    task.metadata["models_used"] = result.get("models_used", [])
    task.metadata["phase_count"] = len(phases)
    if phases:
        last = phases[-1]
        task.metadata["last_model"] = last.get("model")
        task.metadata["last_provider"] = last.get("provider")
    if result.get("status") == "checkpointed":
        task.checkpoint = {
            "version": 3,
            "task_id": str(task.id),
            "active_role": result.get("active_role"),
            "checkpoint_reason": result.get("checkpoint_reason", "timeout"),
            "phases": redact_secrets(phases),
            "messages": redact_secrets(result.get("messages", [])),
            "steps": task.current_step,
            "repairs": task.repair_attempts,
        }
    return result
