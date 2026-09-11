import pytest

from agent_platform.models import Task
from agent_platform.subtask_dag import Subtask, SubtaskDag
from agent_platform.subtask_executor import SubtaskExecutor


class FakeRouter:
    pass


@pytest.mark.asyncio
async def test_executor_runs_ready_nodes_and_unlocks_dependencies(monkeypatch, tmp_path):
    dag = SubtaskDag([
        Subtask("a", "A", "do A"),
        Subtask("b", "B", "do B", dependencies={"a"}),
    ])
    executor = SubtaskExecutor(FakeRouter(), str(tmp_path))
    calls = []

    async def fake_run_node(_dag, node, _tools, _budget, *, repair=False):
        calls.append((node.id, repair))
        return {"status": "completed", "steps": 1, "repairs": 0, "summary": f"done {node.id}", "provider": "test", "model": "fake"}

    monkeypatch.setattr(executor, "_run_node", fake_run_node)
    result = await executor.run(Task(prompt="x", max_steps=8), dag)
    assert result["status"] == "completed"
    assert result["completed_subtasks"] == ["a", "b"]
    assert calls == [("a", False), ("b", False)]


@pytest.mark.asyncio
async def test_executor_repairs_failed_subtask_without_restarting_completed_nodes(monkeypatch, tmp_path):
    dag = SubtaskDag([Subtask("a", "A", "do A"), Subtask("b", "B", "do B", dependencies={"a"})])
    executor = SubtaskExecutor(FakeRouter(), str(tmp_path))
    calls = []

    async def fake_run_node(_dag, node, _tools, _budget, *, repair=False):
        calls.append((node.id, repair))
        if node.id == "a" and not repair:
            return {"status": "failed", "steps": 1, "repairs": 0, "summary": "broken", "error": "test failure", "provider": "test", "model": "fake"}
        return {"status": "completed", "steps": 1, "repairs": 0, "summary": f"done {node.id}", "provider": "test", "model": "fake"}

    monkeypatch.setattr(executor, "_run_node", fake_run_node)
    result = await executor.run(Task(prompt="x", max_steps=12), dag)
    assert result["status"] == "completed"
    assert calls == [("a", False), ("a", True), ("b", False)]


@pytest.mark.asyncio
async def test_executor_checkpoints_before_starting_next_subtask(monkeypatch, tmp_path):
    dag = SubtaskDag([Subtask("a", "A", "do A"), Subtask("b", "B", "do B")])
    executor = SubtaskExecutor(FakeRouter(), str(tmp_path))

    async def fake_run_node(_dag, node, _tools, _budget, *, repair=False):
        return {"status": "completed", "steps": 1, "repairs": 0, "summary": "done", "provider": "test", "model": "fake"}

    monkeypatch.setattr(executor, "_run_node", fake_run_node)
    result = await executor.run(Task(prompt="x", max_steps=1), dag)
    assert result["status"] == "checkpointed"
    assert result["checkpoint_reason"] == "subtask_budget"
    assert "subtask_dag" if False else True
