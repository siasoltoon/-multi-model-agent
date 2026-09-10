from dataclasses import dataclass
from .dag import DagNode, TaskDag
from .models import Task
from .router import SmartRouter

@dataclass
class Plan:
    dag: TaskDag

class Planner:
    def plan(self, task: Task) -> Plan:
        dag = TaskDag()
        dag.add(DagNode("analyze", "architect"))
        dag.add(DagNode("implement", "coder", {"analyze"}))
        dag.add(DagNode("test", "tester", {"implement"}))
        dag.add(DagNode("review", "reviewer", {"test"}))
        return Plan(dag)

class AgentEngine:
    def __init__(self, router: SmartRouter):
        self.router = router
        self.planner = Planner()

    def plan(self, task: Task) -> Plan:
        return self.planner.plan(task)

    def select_endpoint(self, *, min_context: int = 0, tools: bool = True):
        return self.router.choose(min_context=min_context, tools=tools)
