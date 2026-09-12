import pytest

from agent_platform.models import Task
from agent_platform.runner import run_task
from agent_platform.subtask_dag import Subtask, SubtaskDag


class FakeRouter:
    def __init__(self):
        self.restored = None

    def restore_provider_state(self, state):
        self.restored = state

    def snapshot_provider_state(self):
        return {"provider-a": {"failures": 0}}


@pytest.mark.asyncio
async def test_runner_checkpoints_and_resumes_without_replanning(monkeypatch, tmp_path):
    router = FakeRouter()
    task = Task(prompt="resume me", max_steps=8)
    dag = SubtaskDag([
        Subtask("a", "A", "do A", status="completed", result="done A"),
        Subtask("b", "B", "do B"),
    ])
    planning_calls = 0
    execution_calls = 0

    class FakeEngine:
        def __init__(self, _router):
            pass

        async def decompose(self, _task, _context):
            nonlocal planning_calls
            planning_calls += 1
            return SubtaskDag.from_dict(dag.to_dict())

    class FakeExecutor:
        def __init__(self, _router, _workspace):
            pass

        async def run(self, _task, current_dag):
            nonlocal execution_calls
            execution_calls += 1
            if execution_calls == 1:
                assert current_dag.nodes["a"].status == "completed"
                assert current_dag.nodes["b"].status == "pending"
                return {
                    "status": "checkpointed",
                    "checkpoint_reason": "subtask_budget",
                    "dag": current_dag.to_dict(),
                    "history": [{"subtask": "a", "status": "completed"}],
                    "steps": 4,
                    "repairs": 0,
                    "completed_subtasks": ["a"],
                }
            assert current_dag.nodes["a"].status == "completed"
            assert current_dag.nodes["b"].status == "pending"
            current_dag.mark_completed("b", "done B")
            return {
                "status": "completed",
                "dag": current_dag.to_dict(),
                "history": [{"subtask": "b", "status": "completed"}],
                "steps": 6,
                "repairs": 1,
                "completed_subtasks": ["a", "b"],
            }

    monkeypatch.setattr("agent_platform.runner.AgentEngine", FakeEngine)
    monkeypatch.setattr("agent_platform.runner.SubtaskExecutor", FakeExecutor)

    first = await run_task(task, router, str(tmp_path))
    assert first["status"] == "checkpointed"
    assert planning_calls == 1
    assert task.checkpoint["checkpoint_reason"] == "subtask_budget"
    assert task.checkpoint["subtask_dag"]["nodes"]["b"]["status"] == "pending"
    assert task.current_step == 4

    second = await run_task(task, router, str(tmp_path))
    assert second["status"] == "completed"
    assert planning_calls == 1
    assert execution_calls == 2
    assert router.restored == {"provider-a": {"failures": 0}}
    assert task.current_step == 6
    assert task.repair_attempts == 1
    assert task.checkpoint == {}


@pytest.mark.asyncio
async def test_runner_clears_checkpoint_after_terminal_failure(monkeypatch, tmp_path):
    router = FakeRouter()
    task = Task(prompt="fail me", checkpoint={"version": 5, "checkpoint_reason": "old"})

    class FakeEngine:
        def __init__(self, _router):
            pass

        async def decompose(self, _task, _context):
            return SubtaskDag([Subtask("a", "A", "do A")])

    class FakeExecutor:
        def __init__(self, _router, _workspace):
            pass

        async def run(self, _task, _dag):
            return {
                "status": "failed",
                "failed_subtasks": ["a"],
                "dag": {"nodes": {}},
                "history": [],
                "steps": 2,
                "repairs": 2,
            }

    monkeypatch.setattr("agent_platform.runner.AgentEngine", FakeEngine)
    monkeypatch.setattr("agent_platform.runner.SubtaskExecutor", FakeExecutor)

    result = await run_task(task, router, str(tmp_path))
    assert result["status"] == "failed"
    assert task.checkpoint == {}
    assert task.current_step == 2
    assert task.repair_attempts == 2
