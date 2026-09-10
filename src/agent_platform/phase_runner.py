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


ROLE_ORDER = (
    "analysis",
    "architecture",
    "coding",
    "testing",
    "review",
    "security",
    "verification",
)

ROLE_TASK_TYPES = {
    "analysis": "coding",
    "architecture": "coding",
    "coding": "coding",
    "testing": "testing",
    "review": "review",
    "security": "review",
    "verification": "testing",
}

ROLE_TOOLING = {
    "analysis": False,
    "architecture": False,
    "coding": True,
    "testing": True,
    "review": False,
    "security": False,
    "verification": True,
}

ROLE_INSTRUCTIONS = {
    "analysis": "Analyze the request and repository context. Do not edit files. Produce concrete requirements, constraints, risks, and acceptance criteria.",
    "architecture": "Design the implementation approach from the analysis. Do not edit files. Produce a concise implementation plan, affected areas, interfaces, and test strategy.",
    "coding": "Implement the planned changes in the workspace. Inspect before editing, make minimal correct changes, and run focused tests.",
    "testing": "Act as the testing specialist. Inspect the implementation, add or improve tests where needed, run relevant tests, and diagnose failures. Only edit when required to make the test suite meaningful or correct.",
    "review": "Review the resulting implementation as a senior code reviewer. Inspect the diff and tests. Identify correctness, maintainability, regression, and integration issues. Do not edit files unless a clearly necessary small correction is required.",
    "security": "Perform a focused security review of the resulting implementation. Check input handling, secrets, permissions, unsafe execution, dependency/integration risks, and common application vulnerabilities. Do not edit files unless a clearly necessary small correction is required.",
    "verification": "Perform final verification. Inspect git diff/status and run the most relevant tests/checks. Only report completion when the requested behavior is actually verified; otherwise identify the remaining failure precisely.",
}


def _api_key(endpoint) -> str:
    return env_api_key(getattr(endpoint, "api_key_env", None)) or os.getenv(f"{endpoint.provider.upper()}_API_KEY", "") or os.getenv("AGENT_API_KEY", "")


def _build_adapter(endpoint, timeout: float):
    return OpenAICompatibleAdapter(endpoint.base_url, _api_key(endpoint), endpoint.model, timeout=timeout)


def _phase_context(task: Task, role: str, prior: list[dict[str, Any]]) -> str:
    parts = [
        f"Original task:\n{task.prompt}",
        f"Current phase: {role}",
    ]
    for item in prior[-4:]:
        summary = str(item.get("summary", "")).strip()
        if summary:
            parts.append(f"Previous phase {item.get('role')}:\n{summary[-8000:]}")
    return "\n\n".join(parts)


def _summary(result: dict[str, Any]) -> str:
    messages = result.get("messages")
    if isinstance(messages, list):
        chunks = []
        for message in messages[-4:]:
            if isinstance(message, dict) and message.get("content"):
                chunks.append(str(message["content"]))
        if chunks:
            return "\n\n".join(chunks)[-12000:]
    return str(result.get("error") or result.get("status") or "")[-12000:]


class PhaseRunner:
    """Execute an explicit multi-model role pipeline with per-role failover."""

    def __init__(self, router: SmartRouter, workspace: str, *, on_phase: Callable[[dict[str, Any]], None] | None = None):
        self.router = router
        self.workspace = workspace
        self.on_phase = on_phase
        self.request_timeout = float(os.getenv("AGENT_MODEL_REQUEST_TIMEOUT", "180"))
        self.command_timeout = float(os.getenv("AGENT_COMMAND_TIMEOUT", "300"))
        self.task_timeout = float(os.getenv("AGENT_TIMEOUT_SECONDS", "1800"))
        self.max_failover = max(1, int(os.getenv("AGENT_MAX_PROVIDER_FAILOVERS", "3")))

    def _adapter(self, role: str):
        ranked = self.router.ranked(
            min_context=4096,
            tools=ROLE_TOOLING[role],
            task_type=ROLE_TASK_TYPES[role],
            role=role,
        )
        selected = ranked[: self.max_failover]

        def on_failure(endpoint_id, error):
            self.router.mark_failure(endpoint_id, error)

        adapter = FailoverAdapter(
            [(endpoint.id, _build_adapter(endpoint, self.request_timeout)) for endpoint in selected],
            on_failure=on_failure,
        )
        return adapter, selected

    async def run(self, task: Task) -> dict[str, Any]:
        started = monotonic()
        tools = WorkspaceTools(self.workspace, command_timeout=self.command_timeout)
        phases: list[dict[str, Any]] = []
        total_steps = 0
        total_repairs = 0
        prior: list[dict[str, Any]] = []

        for role in ROLE_ORDER:
            if monotonic() - started >= self.task_timeout:
                return {"status": "failed", "error": "multi-model phase pipeline timed out", "phases": phases, "steps": total_steps, "repairs": total_repairs}

            adapter, selected = self._adapter(role)
            system = (
                "You are one specialist in a multi-model software engineering pipeline. "
                "Do not assume another model will finish your responsibilities. "
                + ROLE_INSTRUCTIONS[role]
                + " Never claim work is complete without evidence."
            )
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": _phase_context(task, role, prior)},
            ]
            loop = AgentLoop(
                adapter,
                tools.as_tools(),
                AgentPolicy(max_steps=max(1, task.max_steps // len(ROLE_ORDER)), repair_attempts=6, timeout_seconds=self.task_timeout),
            )
            phase_started = monotonic()
            result = await loop.run(messages, tools.specs())
            elapsed_ms = (monotonic() - phase_started) * 1000.0
            active = next((e for e in selected if e.id == adapter.active_endpoint_id), selected[0])
            if result.get("status") == "completed":
                self.router.mark_success(active.id, latency_ms=elapsed_ms)
            else:
                self.router.mark_failure(active.id, result.get("error", f"{role} phase failed"))

            record = {
                "role": role,
                "status": result.get("status"),
                "model": active.model,
                "provider": active.provider,
                "steps": result.get("steps", 0),
                "repairs": result.get("repairs", 0),
                "latency_ms": round(elapsed_ms, 2),
                "summary": _summary(result),
            }
            phases.append(record)
            prior.append(record)
            total_steps += int(result.get("steps", 0) or 0)
            total_repairs += int(result.get("repairs", 0) or 0)
            if self.on_phase:
                self.on_phase(record)

            if result.get("status") != "completed":
                return {
                    "status": "failed",
                    "error": f"{role} phase failed: {result.get('error', 'unknown error')}",
                    "failed_phase": role,
                    "phases": phases,
                    "steps": total_steps,
                    "repairs": total_repairs,
                }

        return {
            "status": "completed",
            "phases": phases,
            "steps": total_steps,
            "repairs": total_repairs,
            "models_used": [f"{p['provider']}/{p['model']}" for p in phases],
            "roles_completed": [p["role"] for p in phases],
        }
