import asyncio

from agent_platform.adapters import FailoverAdapter, ModelResponse
from agent_platform.agent_loop import AgentLoop, AgentPolicy
from agent_platform.workspace_tools import WorkspaceTools


class FakeAdapter:
    def __init__(self):
        self.calls = 0
        self.histories = []

    async def generate(self, messages, *, tools=None):
        self.calls += 1
        self.histories.append(messages)
        if self.calls == 1:
            return ModelResponse("", {}, {}, [{"id": "call-1", "name": "write_file", "arguments": {"path": "x.txt", "content": "ok"}}])
        return ModelResponse("done", {}, {}, [])


def test_agent_loop_executes_tools_and_preserves_tool_protocol(tmp_path):
    async def run():
        adapter = FakeAdapter()
        tools = WorkspaceTools(str(tmp_path))
        result = await AgentLoop(adapter, tools.as_tools(), AgentPolicy(max_steps=4)).run(
            [{"role": "user", "content": "create x.txt"}], tools.specs()
        )
        return result, adapter
    result, adapter = asyncio.run(run())
    assert result["status"] == "completed"
    assert (tmp_path / "x.txt").read_text() == "ok"
    second_history = adapter.histories[1]
    assistant = next(m for m in second_history if m.get("role") == "assistant" and m.get("tool_calls"))
    tool = next(m for m in second_history if m.get("role") == "tool" and m.get("tool_call_id") == "call-1")
    assert assistant["tool_calls"][0]["id"] == "call-1"
    assert tool["tool_call_id"] == "call-1"


def test_agent_loop_retries_provider_failure_without_consuming_step_budget(tmp_path):
    class RetryAdapter:
        def __init__(self):
            self.calls = 0

        async def generate(self, messages, *, tools=None):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary gateway failure")
            return ModelResponse("done", {}, {}, [])

    async def run():
        adapter = RetryAdapter()
        tools = WorkspaceTools(str(tmp_path))
        result = await AgentLoop(adapter, tools.as_tools(), AgentPolicy(max_steps=1, repair_attempts=2)).run(
            [{"role": "user", "content": "finish the task"}], tools.specs()
        )
        return result, adapter

    result, adapter = asyncio.run(run())
    assert result["status"] == "completed"
    assert result["steps"] == 1
    assert result["repairs"] == 1
    assert adapter.calls == 2


def test_agent_loop_checkpoints_when_step_budget_is_exhausted(tmp_path):
    class ToolOnlyAdapter:
        async def generate(self, messages, *, tools=None):
            return ModelResponse("", {}, {}, [{"id": "call-1", "name": "write_file", "arguments": {"path": "x.txt", "content": "ok"}}])

    async def run():
        tools = WorkspaceTools(str(tmp_path))
        return await AgentLoop(ToolOnlyAdapter(), tools.as_tools(), AgentPolicy(max_steps=1)).run(
            [{"role": "user", "content": "keep working"}], tools.specs()
        )

    result = asyncio.run(run())
    assert result["status"] == "checkpointed"
    assert result["steps"] == 1
    assert result["checkpoint_reason"] == "step_budget"


def test_agent_loop_checkpoints_on_timeout_with_distinct_reason(tmp_path):
    class SlowAdapter:
        async def generate(self, messages, *, tools=None):
            return ModelResponse("done", {}, {}, [])

    async def run():
        tools = WorkspaceTools(str(tmp_path))
        return await AgentLoop(SlowAdapter(), tools.as_tools(), AgentPolicy(max_steps=4, timeout_seconds=0.0)).run(
            [{"role": "user", "content": "finish"}], tools.specs()
        )

    result = asyncio.run(run())
    assert result["status"] == "checkpointed"
    assert result["checkpoint_reason"] == "timeout"


def test_agent_loop_does_not_spend_repair_attempts_on_provider_exhaustion(tmp_path):
    class AlwaysFail:
        async def generate(self, messages, *, tools=None):
            raise RuntimeError("Provider returned error")

    async def run():
        adapter = FailoverAdapter([("bad", AlwaysFail())], quarantine_on_failure=True)
        tools = WorkspaceTools(str(tmp_path))
        return await AgentLoop(adapter, tools.as_tools(), AgentPolicy(max_steps=4, repair_attempts=6)).run(
            [{"role": "user", "content": "finish the task"}], tools.specs()
        )

    result = asyncio.run(run())
    assert result["status"] == "failed"
    assert result["steps"] == 0
    assert result["repairs"] == 0
    assert result["provider_failover_exhausted"] is True


def test_workspace_accepts_common_python_and_chain_commands(tmp_path):
    tools = WorkspaceTools(str(tmp_path))
    result = asyncio.run(tools.run_command({"command": "python3 -c \"print('ok')\" && echo done"}))
    assert result["exit_code"] == 0
    assert "ok" in result["output"]
    assert "done" in result["output"]


def test_workspace_reports_allowed_commands_for_model_repair(tmp_path):
    tools = WorkspaceTools(str(tmp_path))
    try:
        asyncio.run(tools.run_command({"command": "unknown-command --version"}))
        assert False, "unknown command was accepted"
    except ValueError as exc:
        message = str(exc)
        assert "unknown-command" in message
        assert "python3" in message
        assert "pytest" in message


def test_workspace_blocks_escape(tmp_path):
    tools = WorkspaceTools(str(tmp_path))
    try:
        tools.read_file({"path": "../secret.txt"})
        assert False, "workspace escape was allowed"
    except ValueError:
        pass
