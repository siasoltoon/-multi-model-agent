import pytest

from agent_platform.orchestrator import DEFAULT_PHASE_GRAPH, PhaseGraph, PhaseNode


def test_default_graph_is_dependency_ordered():
    completed = set()
    order = []
    for _ in DEFAULT_PHASE_GRAPH.nodes:
        ready = DEFAULT_PHASE_GRAPH.ready(completed)
        role = ready[0]
        order.append(role)
        completed.add(role)
    assert order == [node.role for node in DEFAULT_PHASE_GRAPH.nodes]


def test_verification_can_route_failure_back_to_repair():
    assert DEFAULT_PHASE_GRAPH.repair_target("verification") == "repair"
    assert DEFAULT_PHASE_GRAPH.repair_target("coding") is None


def test_next_phase_respects_dependencies():
    assert DEFAULT_PHASE_GRAPH.next_after("analysis", set()) == "architecture"
    assert DEFAULT_PHASE_GRAPH.next_after("testing", {"analysis", "architecture", "coding"}) == "review"


def test_unknown_dependency_is_rejected():
    with pytest.raises(ValueError, match="unknown phase dependency"):
        PhaseGraph((PhaseNode("coding", ("missing",)),))


def test_dependency_cycle_is_rejected():
    with pytest.raises(ValueError, match="phase dependency cycle"):
        PhaseGraph((PhaseNode("a", ("b",)), PhaseNode("b", ("a",))))
