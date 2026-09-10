from dataclasses import dataclass

from .dag import DagNode, TaskDag
from .models import Task
from .router import SmartRouter


ROLE_PLAN = (
    ("analysis", "analyst", set()),
    ("architecture", "architect", {"analysis"}),
    ("coding", "coder", {"architecture"}),
    ("testing", "tester", {"coding"}),
    ("review", "reviewer", {"testing"}),
    ("security", "security_reviewer", {"review"}),
    ("repair", "repairer", {"security"}),
    ("verification", "verifier", {"repair"}),
)


@dataclass
class Plan:
    dag: TaskDag


class Planner:
    """Build the explicit role DAG consumed by the multi-model executor."""

    def plan(self, task: Task) -> Plan:
        dag = TaskDag()
        for node_id, role, dependencies in ROLE_PLAN:
            dag.add(DagNode(node_id, role, set(dependencies)))
        return Plan(dag)


class AgentEngine:
    def __init__(self, router: SmartRouter):
        self.router = router
        self.planner = Planner()

    def plan(self, task: Task) -> Plan:
        return self.planner.plan(task)

    def select_endpoint(self, *, min_context: int = 0, tools: bool = True):
        return self.router.choose(min_context=min_context, tools=tools)

    def select_role_endpoint(self, role: str, *, min_context: int = 0, tools: bool = True, task_type: str = "coding"):
        return self.router.choose(min_context=min_context, tools=tools, task_type=task_type, role=role)
