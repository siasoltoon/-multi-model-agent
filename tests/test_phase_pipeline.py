from agent_platform.engine import Planner
from agent_platform.models import Task
from agent_platform.phase_runner import PhaseRunner, ROLE_ORDER


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


def test_32_step_budget_protects_execution_critical_phases():
    runner = PhaseRunner.__new__(PhaseRunner)
    budgets = runner._budgets(32)
    assert set(budgets) == set(ROLE_ORDER)
    assert sum(budgets.values()) == 32
    assert budgets["analysis"] >= 4
    assert budgets["coding"] >= 8
    assert budgets["testing"] >= 5
    assert budgets["verification"] >= 5


def test_large_budget_preserves_all_roles():
    runner = PhaseRunner.__new__(PhaseRunner)
    budgets = runner._budgets(64)
    assert set(budgets) == set(ROLE_ORDER)
    assert sum(budgets.values()) == 64
    assert all(budgets[role] >= 3 for role in ROLE_ORDER)


def test_small_budget_is_raised_to_safe_pipeline_minimum():
    runner = PhaseRunner.__new__(PhaseRunner)
    budgets = runner._budgets(1)
    assert sum(budgets.values()) == 32
    assert budgets["analysis"] == 4
    assert budgets["coding"] == 9
