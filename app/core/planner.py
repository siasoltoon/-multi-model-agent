from dataclasses import dataclass, field
from typing import Any


@dataclass
class Node:
    id: str
    role: str
    depends_on: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)


class DAGPlanner:
    """Small deterministic planner; LLM planning can be plugged in later."""

    def plan(self, prompt: str) -> list[Node]:
        return [
            Node("analyze", "architect", payload={"prompt": prompt}),
            Node("implement", "coder", ["analyze"]),
            Node("test", "test_engineer", ["implement"]),
            Node("review", "reviewer", ["test"]),
        ]
