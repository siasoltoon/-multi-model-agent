from agent_platform.engine import Planner
from agent_platform.models import Task
from agent_platform.phase_runner import ROLE_ORDER


def test_planner_builds_specialized_role_dag():
    plan = Planner().plan(Task(prompt="implement a feature"))
    assert list(plan.dag.nodes) == list(ROLE_ORDER)
    assert plan.dag.nodes["coding"].role == "coder"
    assert plan.dag.nodes["testing"].dependencies == {"coding"}
    assert plan.dag.nodes["verification"].dependencies == {"repair"}


def test_role_pipeline_contains_distinct_specialists():
    assert ROLE_ORDER == (
        "analysis",
        "architecture",
        "coding",
        "testing",
        "review",
        "security",
        "repair",
        "verification",
    )
