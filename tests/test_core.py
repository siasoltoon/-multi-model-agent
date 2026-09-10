from agent_platform.dag import DagNode, TaskDag
from agent_platform.router import ModelEndpoint, SmartRouter


def test_dag_rejects_cycle():
    dag = TaskDag()
    dag.add(DagNode("a", "architect"))
    dag.add(DagNode("b", "coder", {"a"}))
    dag.nodes["a"].dependencies.add("b")
    try:
        dag.validate()
        assert False, "cycle was accepted"
    except ValueError as exc:
        assert "cycle" in str(exc).lower()


def test_router_skips_unavailable():
    router = SmartRouter([
        ModelEndpoint("bad", "p", "m", health="RATE_LIMITED"),
        ModelEndpoint("good", "p", "m", health="ONLINE"),
    ])
    assert router.choose().id == "good"
