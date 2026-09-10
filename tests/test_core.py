import pytest
from app.core.models import Health, ModelEndpoint
from app.core.router import SmartRouter
from app.core.planner import DAGPlanner


def test_router_ignores_unavailable_endpoint():
    endpoints = [
        ModelEndpoint("bad", "x", quota_available=False),
        ModelEndpoint("good", "y", quality=.9, task_fit=.9),
    ]
    assert SmartRouter().choose(endpoints).provider == "good"


def test_planner_has_dependencies():
    nodes = DAGPlanner().plan("build a telegram bot")
    assert [n.id for n in nodes] == ["analyze", "implement", "test", "review"]
    assert nodes[-1].depends_on == ["test"]
