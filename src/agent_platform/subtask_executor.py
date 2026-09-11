from __future__ import annotations

import os
from time import monotonic
from typing import Any

from .adapters import FailoverAdapter, OpenAICompatibleAdapter, is_provider_quota_exhausted
from .agent_loop import AgentLoop, AgentPolicy
from .discovery import env_api_key
from .models import Task
from .provider_api_catalog import get_api_contract
from .provider_connections import build_provider_adapter
from .provider_registry import get_provider
from .reliability import redact_secrets
from .router import SmartRouter
from .subtask_dag import Subtask, SubtaskDag
from .workspace_tools import WorkspaceTools

ROLE_TOOLS = {
    "analyst": False,
    "architect": False,
    "coder": True,
    "tester": True,
    "reviewer": False,
    "security": False,
    "repairer": True,
}
ROLE_TASK_TYPES = {
    "analyst": "coding",
    "architect": "coding",
    "coder": "coding",
    "tester": "testing",
    "reviewer": "review",
    "security": "review",
    "repairer": "coding",
}
ROLE_INSTRUCTIONS = {
    "analyst": "Analyze the subtask without editing files. Return concrete requirements and acceptance criteria.",
    "architect": "Design the subtask implementation without editing files. Identify interfaces, sequencing, and tests.",
    "coder": "Implement the subtask in the workspace. Inspect before editing and run focused tests.",
    "tester": "Test the subtask and implementation. Add or improve focused tests and diagnose failures.",
    "reviewer": "Review the subtask result for correctness, regressions, maintainability, and integration issues.",
    "security": "Review the subtask for security risks, unsafe execution, secrets, permissions, and input handling.",
    "repairer": "Repair the subtask. Inspect prior findings and failures, fix verified issues, and rerun relevant tests.",
}
TOOL_GRAMMAR = " Tool commands are direct, non-shell execution: use only allowlisted commands. Safe command strings support && and allowlisted | pipelines. Do not use ;, ||, shell redirection, subshells, backticks, or shell fallbacks."


def _build_adapter(endpoint, timeout: float):
    api_key = env_api_key(getattr(endpoint, "api_key_env", None)) or os.getenv(f"{endpoint.provider.upper()}_API_KEY", "") or os.getenv("AGENT_API_KEY", "")
    definition = get_provider(endpoint.provider)
    if definition.adapter in {"openai", "anthropic", "gemini"} or get_api_contract(endpoint.provider) is not None:
        return build_provider_adapter(endpoint.provider, model=endpoint.model, timeout=timeout, api_key=api_key)
    return OpenAICompatibleAdapter(endpoint.base_url, api_key, endpoint.model, timeout=timeout)


class SubtaskExecutor:
    """Execute ready DAG nodes independently with bounded repair and resumable state."""

    def __init__(self, router: SmartRouter, workspace: str):
        self.router = router
        self.workspace = workspace
        self.request_timeout = float(os.getenv("AGENT_MODEL_REQUEST_TIMEOUT_SECONDS", "180"))
        self.command_timeout = float(os.getenv("AGENT_COMMAND_TIMEOUT", "300"))
        self.max_retries = max(1, int(os.getenv("AGENT_SUBTASK_REPAIR_ATTEMPTS", "2")))
        self.max_failover = max(1, int(os.getenv("AGENT_MAX_PROVIDER_FAILOVERS", "5")))
        self.max_runtime = max(1.0, float(os.getenv("AGENT_SUBTASK_MAX_RUNTIME_SECONDS", "1800")))

    def _adapter(self, role: str):
        selected = self.router.ranked_provider_diverse(
            min_context=4096,
            tools=ROLE_TOOLS[role],
            task_type=ROLE_TASK_TYPES[role],
            role=role,
            max_providers=self.max_failover,
        )
        if not selected:
            raise RuntimeError(f"no zero-cost model endpoint available for subtask role: {role}")

        def on_failure(endpoint_id, error):
            self.router.mark_failure(endpoint_id, error, provider_quota_exhausted=is_provider_quota_exhausted(error))
            return False

        return FailoverAdapter([(e.id, _build_adapter(e, self.request_timeout)) for e in selected], on_failure=on_failure, quarantine_on_failure=True), selected

    @staticmethod
    def _context(dag: SubtaskDag, node: Subtask) -> str:
        deps = []
        for dep in sorted(node.dependencies):
            parent = dag.nodes[dep]
            deps.append(f"Dependency {dep} ({parent.title}):\n{parent.result[-5000:]}")
        return "\n\n".join(deps) or "No completed dependencies."

    async def _run_node(self, dag: SubtaskDag, node: Subtask, tools: WorkspaceTools, budget: int, *, repair: bool = False) -> dict[str, Any]:
        role = "repairer" if repair else node.role
        adapter, selected = self._adapter(role)
        system = ROLE_INSTRUCTIONS[role] + TOOL_GRAMMAR + " Never claim completion without evidence."
        user = (
            f"Subtask: {node.title}\nObjective:\n{node.objective}\n\n"
            f"Dependencies:\n{self._context(dag, node)}\n\n"
            "Work only on this subtask. Preserve correct existing work and verify your result."
        )
        loop = AgentLoop(adapter, tools.as_tools(), AgentPolicy(max_steps=budget, repair_attempts=3, timeout_seconds=self.max_runtime))
        result = await loop.run([{"role": "system", "content": system}, {"role": "user", "content": user}], tools.specs())
        active = next((e for e in selected if e.id == adapter.active_endpoint_id), selected[0])
        if result.get("status") == "completed":
            self.router.mark_success(active.id)
        elif not result.get("provider_failover_exhausted"):
            self.router.mark_failure(active.id, result.get("error", "subtask failed"))
        return {
            "status": result.get("status"),
            "error": result.get("error"),
            "summary": redact_secrets(str(result.get("messages", [])[-2:]))[-10000:],
            "steps": int(result.get("steps", 0) or 0),
            "repairs": int(result.get("repairs", 0) or 0),
            "provider": active.provider,
            "model": active.model,
        }

    async def run(self, task: Task, dag: SubtaskDag) -> dict[str, Any]:
        started = monotonic()
        tools = WorkspaceTools(self.workspace, command_timeout=self.command_timeout)
        completed = [node.id for node in dag.nodes.values() if node.status == "completed"]
        total_steps = int(task.current_step or 0)
        total_repairs = int(task.repair_attempts or 0)
        max_steps = max(1, int(task.max_steps))
        history: list[dict[str, Any]] = []

        while True:
            ready = dag.ready()
            if not ready:
                pending = [n.id for n in dag.nodes.values() if n.status == "pending"]
                failed = [n.id for n in dag.nodes.values() if n.status == "failed"]
                if failed:
                    return {"status": "failed", "failed_subtasks": failed, "pending_subtasks": pending, "dag": dag.to_dict(), "history": history, "steps": total_steps, "repairs": total_repairs}
                if pending:
                    return {"status": "failed", "error": "subtask DAG has no ready nodes", "pending_subtasks": pending, "dag": dag.to_dict(), "history": history, "steps": total_steps, "repairs": total_repairs}
                return {"status": "completed", "dag": dag.to_dict(), "history": history, "steps": total_steps, "repairs": total_repairs, "completed_subtasks": completed}

            for node in ready:
                remaining_steps = max_steps - total_steps
                if monotonic() - started >= self.max_runtime or remaining_steps < 2:
                    return {
                        "status": "checkpointed",
                        "active_subtasks": [n.id for n in dag.ready()],
                        "checkpoint_reason": "subtask_budget",
                        "dag": dag.to_dict(),
                        "history": history,
                        "steps": total_steps,
                        "repairs": total_repairs,
                    }

                remaining_nodes = max(1, len([n for n in dag.nodes.values() if n.status != "completed"]))
                budget = max(2, min(remaining_steps, (remaining_steps + remaining_nodes - 1) // remaining_nodes))
                result = await self._run_node(dag, node, tools, budget)
                total_steps += result["steps"]
                total_repairs += result["repairs"]
                history.append({"subtask": node.id, **result})
                if result["status"] == "completed":
                    dag.mark_completed(node.id, result["summary"])
                    completed.append(node.id)
                    continue

                repaired = False
                for attempt in range(self.max_retries):
                    remaining_steps = max_steps - total_steps
                    if remaining_steps < 2 or monotonic() - started >= self.max_runtime:
                        return {
                            "status": "checkpointed",
                            "active_subtasks": [node.id],
                            "checkpoint_reason": "subtask_budget",
                            "dag": dag.to_dict(),
                            "history": history,
                            "steps": total_steps,
                            "repairs": total_repairs,
                        }
                    node.attempts += 1
                    repair_budget = max(2, min(remaining_steps, max(2, budget // 2)))
                    repair = await self._run_node(dag, node, tools, repair_budget, repair=True)
                    total_steps += repair["steps"]
                    total_repairs += repair["repairs"] + 1
                    history.append({"subtask": node.id, "repair_attempt": attempt + 1, **repair})
                    if repair["status"] == "completed":
                        dag.mark_completed(node.id, repair["summary"])
                        completed.append(node.id)
                        repaired = True
                        break
                if not repaired:
                    dag.mark_failed(node.id, result.get("error") or "subtask execution failed")
                    return {"status": "failed", "failed_subtasks": [node.id], "dag": dag.to_dict(), "history": history, "steps": total_steps, "repairs": total_repairs}
