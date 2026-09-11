from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class PhaseNode:
    """A deterministic phase contract used by the task orchestrator."""

    role: str
    depends_on: tuple[str, ...] = ()
    retry_target: str | None = None


class PhaseGraph:
    """Dependency-aware execution graph for resumable coding tasks."""

    def __init__(self, nodes: Iterable[PhaseNode]):
        self.nodes = tuple(nodes)
        self._by_role = {node.role: node for node in self.nodes}
        if len(self._by_role) != len(self.nodes):
            raise ValueError("duplicate phase role")
        self._validate()

    def _validate(self) -> None:
        for node in self.nodes:
            for dependency in node.depends_on:
                if dependency not in self._by_role:
                    raise ValueError(f"unknown phase dependency: {dependency}")
            if node.retry_target is not None and node.retry_target not in self._by_role:
                raise ValueError(f"unknown retry target: {node.retry_target}")
        self._assert_acyclic()

    def _assert_acyclic(self) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(role: str) -> None:
            if role in visiting:
                raise ValueError("phase dependency cycle detected")
            if role in visited:
                return
            visiting.add(role)
            for dependency in self._by_role[role].depends_on:
                visit(dependency)
            visiting.remove(role)
            visited.add(role)

        for node in self.nodes:
            visit(node.role)

    def node(self, role: str) -> PhaseNode:
        try:
            return self._by_role[role]
        except KeyError as exc:
            raise ValueError(f"unknown phase role: {role}") from exc

    def ready(self, completed: set[str], *, active: str | None = None) -> list[str]:
        return [
            node.role
            for node in self.nodes
            if node.role not in completed
            and (active is None or node.role == active)
            and all(dependency in completed for dependency in node.depends_on)
        ]

    def next_after(self, role: str, completed: set[str]) -> str | None:
        if role not in self._by_role:
            raise ValueError(f"unknown phase role: {role}")
        effective_completed = set(completed) | {role}
        for candidate in self.nodes:
            if candidate.role in effective_completed:
                continue
            if all(dependency in effective_completed for dependency in candidate.depends_on):
                return candidate.role
        return None

    def repair_target(self, role: str) -> str | None:
        return self.node(role).retry_target


DEFAULT_PHASE_GRAPH = PhaseGraph(
    (
        PhaseNode("analysis"),
        PhaseNode("architecture", ("analysis",)),
        PhaseNode("coding", ("architecture",)),
        PhaseNode("testing", ("coding",)),
        PhaseNode("review", ("testing",)),
        PhaseNode("security", ("review",)),
        PhaseNode("repair", ("security",), retry_target="coding"),
        PhaseNode("verification", ("repair",), retry_target="repair"),
    )
)
