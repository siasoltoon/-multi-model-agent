import asyncio

from agent_platform.adapters import ModelResponse
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
    assert second_history[-2]["role"] == "assistant"
    assert second_history[-2]["tool_calls"][0]["id"] == "call-1"
    assert second_history[-1]["role"] == "tool"
    assert second_history[-1]["tool_call_id"] == "call-1"


def test_workspace_blocks_escape(tmp_path):
    tools = WorkspaceTools(str(tmp_path))
    try:
        tools.read_file({"path": "../secret.txt"})
        assert False, "workspace escape was allowed"
    except ValueError:
        pass
