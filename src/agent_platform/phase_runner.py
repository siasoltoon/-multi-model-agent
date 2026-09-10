from __future__ import annotations

import os
from time import monotonic
from typing import Any, Callable

from .adapters import FailoverAdapter, OpenAICompatibleAdapter
from .agent_loop import AgentLoop, AgentPolicy
from .discovery import env_api_key
from .models import Task
from .router import SmartRouter
from .workspace_tools import WorkspaceTools

ROLE_ORDER = ("analysis", "architecture", "coding", "testing", "review", "security", "repair", "verification")
ROLE_TASK_TYPES = {"analysis": "coding", "architecture": "coding", "coding": "coding", "testing": "testing", "review": "review", "security": "review", "repair": "coding", "verification": "testing"}
ROLE_TOOLING = {"analysis": False, "architecture": False, "coding": True, "testing": True, "review": False, "security": False, "repair": True, "verification": True}
ROLE_WEIGHTS = {"analysis": 4, "architecture": 6, "coding": 20, "testing": 12, "review": 7, "security": 5, "repair": 8, "verification": 8}
ROLE_INSTRUCTIONS = {
    "analysis": "Analyze the request and repository context. Do not edit files. Produce requirements, constraints, risks, and acceptance criteria.",
    "architecture": "Design the implementation approach from the analysis. Do not edit files. Produce affected areas, interfaces, sequencing, and test strategy.",
    "coding": "Implement the planned changes in the workspace. Inspect before editing, make minimal correct changes, and run focused tests.",
    "testing": "Act as the testing specialist. Inspect the implementation, add or improve tests where needed, run relevant tests, and diagnose failures.",
    "review": "Review the resulting implementation as a senior reviewer. Inspect the diff and tests. Identify correctness, maintainability, regression, and integration issues.",
    "security": "Perform a focused security review. Check input handling, secrets, permissions, unsafe execution, dependency/integration risks, and common vulnerabilities.",
    "repair": "Act as the repair specialist. Inspect the current workspace, previous findings, failing tests, and diff. Fix every concrete issue you can verify, then rerun relevant tests.",
    "verification": "Perform final verification. Inspect git diff/status and run the most relevant tests/checks. Only report completion when the requested behavior is actually verified.",
}


def _api_key(endpoint) -> str:
    return env_api_key(getattr(endpoint, "api_key_env", None)) or os.getenv(f"{endpoint.provider.upper()}_API_KEY", "") or os.getenv("AGENT_API_KEY", "")


def _build_adapter(endpoint, timeout: float):
    return OpenAICompatibleAdapter(endpoint.base_url, _api_key(endpoint), endpoint.model, timeout=timeout)


def _phase_context(task: Task, role: str, prior: list[dict[str, Any]]) -> str:
    parts = [f"Original task:\n{task.prompt}", f"Current phase: {role}"]
    for item in prior[-5:]:
        summary = str(item.get("summary", "")).strip()
        if summary:
            parts.append(f"Previous phase {item.get('role')}:\n{summary[-6000:]}")
    return "\n\n".join(parts)


def _summary(result: dict[str, Any]) -> str:
    messages = result.get("messages")
    if isinstance(messages, list):
        chunks = [str(m["content"]) for m in messages[-4:] if isinstance(m, dict) and m.get("content")]
        if chunks:
            return "\n\n".join(chunks)[-10000:]
    return str(result.get("error") or result.get("status") or "")[-10000:]


class PhaseRunner:
    """Execute the complete role pipeline with independent routing and failover per phase."""

    def __init__(self, router: SmartRouter, workspace: str, *, on_phase: Callable[[dict[str, Any]], None] | None = None):
        self.router = router
        self.workspace = workspace
        self.on_phase = on_phase
        self.request_timeout = float(os.getenv("AGENT_MODEL_REQUEST_TIMEOUT", "180"))
        self.command_timeout = float(os.getenv("AGENT_COMMAND_TIMEOUT", "300"))
        self.task_timeout = float(os.getenv("AGENT_TIMEOUT_SECONDS", "1800"))
        self.max_failover = max(1, int(os.getenv("AGENT_MAX_PROVIDER_FAILOVERS", "3")))

    def _adapter(self, role: str):
        ranked = self.router.ranked(min_context=4096, tools=ROLE_TOOLING[role], task_type=ROLE_TASK_TYPES[role], role=role)
        selected = ranked[: self.max_failover]
        adapter = FailoverAdapter([(e.id, _build_adapter(e, self.request_timeout)) for e in selected], on_failure=lambda endpoint_id, error: self.router.mark_failure(endpoint_id, error))
        return adapter, selected

    def _budget(self, role: str, total: int) -> int:
        scale = max(0.25, total / 70.0)
        return max(2, min(24, round(ROLE_WEIGHTS[role] * scale)))

    async def run(self, task: Task) -> dict[str, Any]:
        started = monotonic()
        tools = WorkspaceTools(self.workspace, command_timeout=self.command_timeout)
        phases: list[dict[str, Any]] = []
        prior: list[dict[str, Any]] = []
        total_steps = 0
        total_repairs = 0

        for role in ROLE_ORDER:
            if monotonic() - started >= self.task_timeout:
                return {"status": "failed", "error": "multi-model phase pipeline timed out", "phases": phases, "steps": total_steps, "repairs": total_repairs}
            adapter, selected = self._adapter(role)
            messages = [
                {"role": "system", "content": "You are one specialist in an automatic multi-model software engineering pipeline. " + ROLE_INSTRUCTIONS[role] + " Never claim completion without evidence."},
                {"role": "user", "content": _phase_context(task, role, prior)},
            ]
            loop = AgentLoop(adapter, tools.as_tools(), AgentPolicy(max_steps=self._budget(role, task.max_steps), repair_attempts=6, timeout_seconds=self.task_timeout))
            phase_started = monotonic()
            result = await loop.run(messages, tools.specs())
            elapsed_ms = (monotonic() - phase_started) * 1000.0
            active = next((e for e in selected if e.id == adapter.active_endpoint_id), selected[0])
            if result.get("status") == "completed":
                self.router.mark_success(active.id, latency_ms=elapsed_ms)
            else:
                self.router.mark_failure(active.id, result.get("error", f"{role} phase failed"))
            record = {"role": role, "status": result.get("status"), "model": active.model, "provider": active.provider, "steps": result.get("steps", 0), "repairs": result.get("repairs", 0), "latency_ms": round(elapsed_ms, 2), "summary": _summary(result)}
            phases.append(record)
            prior.append(record)
            total_steps += int(result.get("steps", 0) or 0)
            total_repairs += int(result.get("repairs", 0) or 0)
            if self.on_phase:
                self.on_phase(record)
            if result.get("status") != "completed":
                return {"status": "failed", "error": f"{role} phase failed: {result.get('error', 'unknown error')}", "failed_phase": role, "phases": phases, "steps": total_steps, "repairs": total_repairs}

        return {"status": "completed", "phases": phases, "steps": total_steps, "repairs": total_repairs, "models_used": [f"{p['provider']}/{p['model']}" for p in phases], "roles_completed": [p["role"] for p in phases]}
