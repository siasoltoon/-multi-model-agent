from __future__ import annotations

from .models import Task
from .phase_runner import PhaseRunner
from .router import SmartRouter


async def run_task(task: Task, router: SmartRouter, workspace: str) -> dict:
    """Run a task through the automatic role-based multi-model pipeline."""
    result = await PhaseRunner(router, workspace).run(task)
    task.current_step = result.get("steps", task.current_step)
    task.repair_attempts = result.get("repairs", task.repair_attempts)
    task.result = result
    phases = result.get("phases", [])
    task.metadata["execution_mode"] = "role_based_multi_model"
    task.metadata["roles_completed"] = [p.get("role") for p in phases]
    task.metadata["models_used"] = result.get("models_used", [])
    task.metadata["phase_count"] = len(phases)
    if phases:
        last = phases[-1]
        task.metadata["last_model"] = last.get("model")
        task.metadata["last_provider"] = last.get("provider")
    if result.get("status") == "checkpointed":
        task.checkpoint = {"phases": phases, "steps": task.current_step, "repairs": task.repair_attempts}
    return result
