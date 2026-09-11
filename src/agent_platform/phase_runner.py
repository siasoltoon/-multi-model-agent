from __future__ import annotations

import os
from time import monotonic
from typing import Any, Callable

from .adapters import FailoverAdapter, OpenAICompatibleAdapter
from .agent_loop import AgentLoop, AgentPolicy
from .discovery import env_api_key
from .models import Task
from .provider_api_catalog import get_api_contract
from .provider_connections import build_provider_adapter
from .provider_registry import get_provider
from .router import SmartRouter
from .reliability import redact_secrets
from .workspace_tools import WorkspaceTools

ROLE_ORDER = ("analysis", "architecture", "coding", "testing", "review", "security", "repair", "verification")
ROLE_TASK_TYPES = {"analysis": "coding", "architecture": "coding", "coding": "coding", "testing": "testing", "review": "review", "security": "review", "repair": "coding", "verification": "testing"}
ROLE_TOOLING = {"analysis": False, "architecture": False, "coding": True, "testing": True, "review": False, "security": False, "repair": True, "verification": True}
ROLE_WEIGHTS = {"analysis": 4, "architecture": 6, "coding": 20, "testing": 12, "review": 7, "security": 5, "repair": 8, "verification": 8}
ROLE_INSTRUCTIONS = {
    "analysis": "Analyze the request and repository context. Do not edit files. Inspect safely with read_file/run_command when useful. Produce requirements, constraints, risks, and acceptance criteria.",
    "architecture": "Design the implementation approach from the analysis. Do not edit files. Inspect safely when useful. Produce affected areas, interfaces, sequencing, and test strategy.",
    "coding": "Implement the planned changes in the workspace. Inspect before editing, make minimal correct changes, and run focused tests.",
    "testing": "Act as the testing specialist. Inspect the implementation, add or improve tests where needed, run relevant tests, and diagnose failures.",
    "review": "Review the resulting implementation as a senior reviewer. Inspect the diff and tests. Identify correctness, maintainability, regression, and integration issues.",
    "security": "Perform a focused security review. Check input handling, secrets, permissions, unsafe execution, dependency/integration risks, and common vulnerabilities.",
    "repair": "Act as the repair specialist. Inspect the current workspace, previous findings, failing tests, and diff. Fix every concrete issue you can verify, then rerun relevant tests.",
    "verification": "Perform final verification. Inspect git diff/status and run the most relevant tests/checks. Only report completion when the requested behavior is actually verified.",
}
TOOL_GRAMMAR = " Tool commands are direct, non-shell execution: use only allowlisted commands. Safe command strings support && and allowlisted | pipelines. Do not use ;, ||, shell redirection, subshells, backticks, or shell fallbacks; issue a separate tool call instead."


def _api_key(endpoint) -> str:
    return env_api_key(getattr(endpoint, "api_key_env", None)) or os.getenv(f"{endpoint.provider.upper()}_API_KEY", "") or os.getenv("AGENT_API_KEY", "")


def _build_adapter(endpoint, timeout: float):
    definition = get_provider(endpoint.provider)
    if definition.adapter in {"openai", "anthropic", "gemini"}:
        return build_provider_adapter(endpoint.provider, model=endpoint.model, timeout=timeout, api_key=_api_key(endpoint))
    if get_api_contract(endpoint.provider) is not None:
        return build_provider_adapter(endpoint.provider, model=endpoint.model, timeout=timeout, api_key=_api_key(endpoint))
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


def _checkpoint_phases(task: Task) -> tuple[list[dict[str, Any]], str | None, list[dict[str, Any]]]:
    checkpoint = task.checkpoint or {}
    raw = checkpoint.get("phases", []) if isinstance(checkpoint, dict) else []
    phases = [dict(item) for item in raw if isinstance(item, dict) and item.get("role") in ROLE_ORDER]
    active_role = checkpoint.get("active_role") if isinstance(checkpoint, dict) else None
    messages = checkpoint.get("messages", []) if isinstance(checkpoint, dict) else []
    return phases, active_role if active_role in ROLE_ORDER else None, messages if isinstance(messages, list) else []


class PhaseRunner:
    """Execute the role pipeline with provider failover and resumable role checkpoints."""

    def __init__(self, router: SmartRouter, workspace: str, *, on_phase: Callable[[dict[str, Any]], None] | None = None):
        self.router = router
        self.workspace = workspace
        self.on_phase = on_phase
        self.request_timeout = float(os.getenv("AGENT_MODEL_REQUEST_TIMEOUT_SECONDS", os.getenv("AGENT_MODEL_REQUEST_TIMEOUT", "180")))
        self.command_timeout = float(os.getenv("AGENT_COMMAND_TIMEOUT", "300"))
        self.task_timeout = float(os.getenv("AGENT_TIMEOUT_SECONDS", "1800"))
        self.max_failover = max(1, int(os.getenv("AGENT_MAX_PROVIDER_FAILOVERS", "5")))

    def _adapter(self, role: str):
        ranked = self.router.ranked(min_context=4096, tools=ROLE_TOOLING[role], task_type=ROLE_TASK_TYPES[role], role=role)
        selected = ranked[: self.max_failover]
        if not selected:
            raise RuntimeError(f"no model endpoint available for role: {role}")
        adapter = FailoverAdapter(
            [(e.id, _build_adapter(e, self.request_timeout)) for e in selected],
            on_failure=lambda endpoint_id, error: self.router.mark_failure(endpoint_id, error),
            quarantine_on_failure=True,
        )
        return adapter, selected

    def _budgets(self, total: int) -> dict[str, int]:
        """Allocate the global budget so a 32-step run can actually reach coding and verification."""
        total = max(32, int(total))
        if total == 32:
            values = {"analysis": 4, "architecture": 3, "coding": 9, "testing": 5, "review": 2, "security": 1, "repair": 2, "verification": 6}
            if sum(values.values()) != 32:
                raise AssertionError("invalid 32-step role budget")
            return values

        minimum = 3
        remaining = total - minimum * len(ROLE_ORDER)
        weights = [ROLE_WEIGHTS[role] for role in ROLE_ORDER]
        weight_sum = sum(weights)
        raw = [remaining * weight / weight_sum for weight in weights]
        extras = [int(value) for value in raw]
        remainder = remaining - sum(extras)
        order = sorted(range(len(ROLE_ORDER)), key=lambda index: raw[index] - extras[index], reverse=True)
        for index in order[:remainder]:
            extras[index] += 1
        return {role: minimum + extras[index] for index, role in enumerate(ROLE_ORDER)}

    async def run(self, task: Task) -> dict[str, Any]:
        started = monotonic()
        tools = WorkspaceTools(self.workspace, command_timeout=self.command_timeout)
        phases, active_role, checkpoint_messages = _checkpoint_phases(task)
        completed_roles = {p["role"] for p in phases if p.get("status") == "completed"}
        prior = list(phases)
        total_steps = int(task.current_step or sum(int(p.get("steps", 0) or 0) for p in phases))
        total_repairs = int(task.repair_attempts or sum(int(p.get("repairs", 0) or 0) for p in phases))
        resume_messages = checkpoint_messages if active_role else []
        budgets = self._budgets(task.max_steps)

        for role in ROLE_ORDER:
            if role in completed_roles and role != active_role:
                continue
            if monotonic() - started >= self.task_timeout:
                return {"status": "checkpointed", "active_role": role, "error": "multi-model phase pipeline timed out", "phases": phases, "steps": total_steps, "repairs": total_repairs}
            adapter, selected = self._adapter(role)
            if role == active_role and resume_messages:
                messages = resume_messages
            else:
                messages = [
                    {"role": "system", "content": "You are one specialist in an automatic multi-model software engineering pipeline. " + ROLE_INSTRUCTIONS[role] + TOOL_GRAMMAR + " Never claim completion without evidence."},
                    {"role": "user", "content": _phase_context(task, role, prior)},
                ]
            loop = AgentLoop(adapter, tools.as_tools(), AgentPolicy(max_steps=budgets[role], repair_attempts=6, timeout_seconds=self.task_timeout))
            phase_started = monotonic()
            result = await loop.run(messages, tools.specs())
            elapsed_ms = (monotonic() - phase_started) * 1000.0
            active = next((e for e in selected if e.id == adapter.active_endpoint_id), selected[0])
            if result.get("status") == "completed":
                self.router.mark_success(active.id, latency_ms=elapsed_ms)
            else:
                self.router.mark_failure(active.id, result.get("error", f"{role} phase failed"))
            record = {
                "role": role, "status": result.get("status"), "model": active.model, "provider": active.provider,
                "steps": result.get("steps", 0), "repairs": result.get("repairs", 0), "latency_ms": round(elapsed_ms, 2),
                "summary": _summary(result),
            }
            phases = [p for p in phases if p.get("role") != role]
            phases.append(record)
            prior = list(phases)
            total_steps += int(result.get("steps", 0) or 0)
            total_repairs += int(result.get("repairs", 0) or 0)
            if self.on_phase:
                self.on_phase(record)
            if result.get("status") == "checkpointed":
                return {
                    "status": "checkpointed", "active_role": role, "phases": phases, "steps": total_steps, "repairs": total_repairs,
                    "messages": redact_secrets(result.get("messages", [])),
                }
            if result.get("status") != "completed":
                return {"status": "failed", "error": f"{role} phase failed: {result.get('error', 'unknown error')}", "failed_phase": role, "phases": phases, "steps": total_steps, "repairs": total_repairs}
            completed_roles.add(role)
            active_role = None
            resume_messages = []
        return {"status": "completed", "phases": phases, "steps": total_steps, "repairs": total_repairs, "models_used": [f"{p['provider']}/{p['model']}" for p in phases], "roles_completed": [p["role"] for p in phases]}
