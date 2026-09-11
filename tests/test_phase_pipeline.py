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


def test_phase_budgets_never_starve_a_role():
    runner = PhaseRunner.__new__(PhaseRunner)
    for total in (32, 64):
        budgets = runner._budgets(total)
        assert set(budgets) == set(ROLE_ORDER)
        assert sum(budgets.values()) == total
        assert all(budgets[role] >= 3 for role in ROLE_ORDER)


def test_small_budget_is_raised_to_safe_pipeline_minimum():
    runner = PhaseRunner.__new__(PhaseRunner)
    budgets = runner._budgets(1)
    assert sum(budgets.values()) == len(ROLE_ORDER) * 3
    assert all(budgets[role] == 3 for role in ROLE_ORDER)
