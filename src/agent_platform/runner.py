from __future__ import annotations

from .engine import AgentEngine
from .models import Task
from .reliability import redact_secrets
from .router import SmartRouter
from .subtask_dag import SubtaskDag
from .subtask_executor import SubtaskExecutor


async def run_task(task: Task, router: SmartRouter, workspace: str) -> dict:
    """Run a task through decomposition, DAG-aware execution, repair, and resume."""
    prior_state = task.checkpoint.get("provider_state") if isinstance(task.checkpoint, dict) else None
    if prior_state:
        router.restore_provider_state(prior_state)

    checkpoint_dag = task.checkpoint.get("subtask_dag") if isinstance(task.checkpoint, dict) else None
    if checkpoint_dag:
        dag = SubtaskDag.from_dict(checkpoint_dag)
    else:
        context = str(task.checkpoint.get("planning_context", "")) if isinstance(task.checkpoint, dict) else ""
        dag = await AgentEngine(router).decompose(task, context)

    result = await SubtaskExecutor(router, workspace).run(task, dag)
    task.current_step = int(result.get("steps", task.current_step) or 0)
    task.repair_attempts = int(result.get("repairs", task.repair_attempts) or 0)
    task.result = redact_secrets(result)
    task.metadata["execution_mode"] = "decomposed_subtask_dag"
    task.metadata["subtask_count"] = len(dag.nodes)
    task.metadata["completed_subtasks"] = result.get("completed_subtasks", [])
    task.metadata["failed_subtasks"] = result.get("failed_subtasks", [])
    task.metadata["subtask_history"] = redact_secrets(result.get("history", []))

    if result.get("status") == "checkpointed":
        task.checkpoint = {
            "version": 5,
            "task_id": str(task.id),
            "checkpoint_reason": result.get("checkpoint_reason", "subtask_budget"),
            "subtask_dag": redact_secrets(result.get("dag", dag.to_dict())),
            "provider_state": redact_secrets(router.snapshot_provider_state()),
            "steps": task.current_step,
            "repairs": task.repair_attempts,
        }
    elif result.get("status") in {"completed", "failed"}:
        task.checkpoint = {}

    return result
