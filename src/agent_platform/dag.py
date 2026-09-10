from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class DagNode:
    id: str
    role: str
    dependencies: set[str] = field(default_factory=set)
    status: str = "pending"

@dataclass
class TaskDag:
    nodes: dict[str, DagNode] = field(default_factory=dict)

    def add(self, node: DagNode) -> None:
        if node.id in self.nodes:
            raise ValueError(f"duplicate node: {node.id}")
        self.nodes[node.id] = node
        self.validate()

    def validate(self) -> None:
        for node in self.nodes.values():
            if node.dependencies - self.nodes.keys():
                raise ValueError(f"unknown dependency for {node.id}")
        visiting, visited = set(), set()
        def visit(key: str):
            if key in visiting:
                raise ValueError("DAG cycle detected")
            if key in visited:
                return
            visiting.add(key)
            for dep in self.nodes[key].dependencies:
                visit(dep)
            visiting.remove(key); visited.add(key)
        for key in self.nodes:
            visit(key)

    def ready(self) -> list[DagNode]:
        return [n for n in self.nodes.values() if n.status == "pending" and all(self.nodes[d].status == "completed" for d in n.dependencies)]
