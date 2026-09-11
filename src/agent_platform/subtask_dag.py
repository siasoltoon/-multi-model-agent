from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Subtask:
    id: str
    title: str
    objective: str
    role: str = "coder"
    dependencies: set[str] = field(default_factory=set)
    status: str = "pending"
    attempts: int = 0
    result: str = ""


class SubtaskDag:
    """Validated task-level DAG with resumable node state."""

    def __init__(self, subtasks: list[Subtask] | None = None):
        self.nodes: dict[str, Subtask] = {}
        for subtask in subtasks or []:
            self.add(subtask)

    def add(self, subtask: Subtask) -> None:
        if not subtask.id or subtask.id in self.nodes:
            raise ValueError(f"duplicate subtask: {subtask.id}")
        self.nodes[subtask.id] = subtask
        try:
            self.validate()
        except Exception:
            del self.nodes[subtask.id]
            raise

    def validate(self) -> None:
        known = set(self.nodes)
        for node in self.nodes.values():
            missing = node.dependencies - known
            if missing:
                raise ValueError(f"unknown subtask dependency for {node.id}: {sorted(missing)}")
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in visiting:
                raise ValueError("subtask DAG cycle detected")
            if node_id in visited:
                return
            visiting.add(node_id)
            for dependency in self.nodes[node_id].dependencies:
                visit(dependency)
            visiting.remove(node_id)
            visited.add(node_id)

        for node_id in self.nodes:
            visit(node_id)

    def ready(self) -> list[Subtask]:
        return [
            node for node in self.nodes.values()
            if node.status == "pending"
            and all(self.nodes[dep].status == "completed" for dep in node.dependencies)
        ]

    def mark_completed(self, node_id: str, result: str = "") -> None:
        node = self.nodes[node_id]
        node.status = "completed"
        node.result = result

    def mark_failed(self, node_id: str, result: str = "") -> None:
        node = self.nodes[node_id]
        node.status = "failed"
        node.result = result

    def to_dict(self) -> dict[str, Any]:
        return {"nodes": {
            node_id: {
                "id": node.id,
                "title": node.title,
                "objective": node.objective,
                "role": node.role,
                "dependencies": sorted(node.dependencies),
                "status": node.status,
                "attempts": node.attempts,
                "result": node.result[-10000:],
            }
            for node_id, node in self.nodes.items()
        }}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubtaskDag":
        nodes = data.get("nodes", {}) if isinstance(data, dict) else {}
        if not isinstance(nodes, dict):
            raise ValueError("invalid subtask DAG payload")
        return cls([
            Subtask(
                id=str(node_id),
                title=str(value.get("title", node_id)),
                objective=str(value.get("objective", "")),
                role=str(value.get("role", "coder")),
                dependencies=set(value.get("dependencies", [])),
                status=str(value.get("status", "pending")),
                attempts=int(value.get("attempts", 0) or 0),
                result=str(value.get("result", "")),
            )
            for node_id, value in nodes.items()
            if isinstance(value, dict)
        ])


def parse_subtask_plan(text: str, *, max_subtasks: int = 12) -> SubtaskDag:
    """Parse strict JSON from a planner; markdown fences are tolerated."""
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
        raw = raw.rsplit("```", 1)[0].strip()
    payload = json.loads(raw)
    items = payload.get("subtasks") if isinstance(payload, dict) else None
    if not isinstance(items, list) or not items:
        raise ValueError("planner returned no subtasks")
    if len(items) > max_subtasks:
        raise ValueError(f"planner returned too many subtasks: {len(items)}")
    result = SubtaskDag()
    for index, item in enumerate(items, 1):
        if not isinstance(item, dict):
            raise ValueError(f"invalid subtask at index {index}")
        node_id = str(item.get("id", "")).strip()
        title = str(item.get("title", "")).strip()
        objective = str(item.get("objective", "")).strip()
        role = str(item.get("role", "coder")).strip().lower()
        if not node_id or not title or not objective:
            raise ValueError(f"incomplete subtask at index {index}")
        if role not in {"analyst", "architect", "coder", "tester", "reviewer", "security", "repairer"}:
            raise ValueError(f"unsupported subtask role: {role}")
        dependencies = {str(value).strip() for value in item.get("dependencies", []) if str(value).strip()}
        result.add(Subtask(node_id, title, objective, role, dependencies))
    result.validate()
    return result


PLANNER_INSTRUCTIONS = """Decompose the software task into the smallest useful independently executable subtasks.
Return ONLY valid JSON with this shape:
{"subtasks":[{"id":"s1","title":"...","objective":"...","role":"coder","dependencies":[]}]}
Use 1-12 subtasks. IDs must be short and unique. Dependencies must reference existing IDs.
Roles: analyst, architect, coder, tester, reviewer, security, repairer.
Prefer independent work where safe, but make dependencies explicit. Do not invent requirements.
"""


class SubtaskPlanner:
    """Model-backed decomposition with strict validation and bounded output."""

    def __init__(self, adapter, *, max_subtasks: int = 12):
        self.adapter = adapter
        self.max_subtasks = max(1, min(max_subtasks, 12))

    async def plan(self, task_prompt: str, context: str = "") -> SubtaskDag:
        response = await self.adapter.generate([
            {"role": "system", "content": PLANNER_INSTRUCTIONS},
            {"role": "user", "content": f"Task:\n{task_prompt}\n\nRepository context:\n{context[-12000:]}"},
        ])
        return parse_subtask_plan(response.text, max_subtasks=self.max_subtasks)
