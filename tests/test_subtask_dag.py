import json

import pytest

from agent_platform.subtask_dag import Subtask, SubtaskDag, parse_subtask_plan


def test_ready_respects_dependencies():
    dag = SubtaskDag([Subtask("a", "A", "do A"), Subtask("b", "B", "do B", dependencies={"a"})])
    assert [node.id for node in dag.ready()] == ["a"]
    dag.mark_completed("a", "done")
    assert [node.id for node in dag.ready()] == ["b"]


def test_cycle_is_rejected():
    with pytest.raises(ValueError, match="subtask DAG cycle"):
        SubtaskDag([Subtask("a", "A", "", dependencies={"b"}), Subtask("b", "B", "", dependencies={"a"})])


def test_unknown_dependency_is_rejected():
    with pytest.raises(ValueError, match="unknown subtask dependency"):
        SubtaskDag([Subtask("a", "A", "", dependencies={"missing"})])


def test_parse_plan_accepts_json_fence_and_round_trips():
    text = "```json\n" + json.dumps({"subtasks": [
        {"id": "s1", "title": "API", "objective": "Implement API", "role": "coder", "dependencies": []},
        {"id": "s2", "title": "Tests", "objective": "Test API", "role": "tester", "dependencies": ["s1"]},
    ]}) + "\n```"
    dag = parse_subtask_plan(text)
    assert [node.id for node in dag.ready()] == ["s1"]
    restored = SubtaskDag.from_dict(dag.to_dict())
    assert restored.nodes["s2"].dependencies == {"s1"}


def test_parse_plan_rejects_unknown_role():
    with pytest.raises(ValueError, match="unsupported subtask role"):
        parse_subtask_plan(json.dumps({"subtasks": [{"id": "s1", "title": "x", "objective": "x", "role": "manager"}]}))
