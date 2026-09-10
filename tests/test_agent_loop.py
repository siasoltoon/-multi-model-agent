import asyncio

import pytest

from agent_platform.adapters import ModelResponse
from agent_platform.agent_loop import AgentLoop, AgentPolicy
from agent_platform.workspace_tools import WorkspaceTools


class FakeAdapter:
    def __init__(self):
        self.calls = 0

    async def generate(self, messages, *, tools=None):
        self.calls += 1
        if self.calls == 1:
            return ModelResponse("", {}, {}, [{"name": "write_file", "arguments": {"path": "x.txt", "content": "ok"}}])
        return ModelResponse("done", {}, {}, [])


def test_agent_loop_executes_tools(tmp_path):
    async def run():
        tools = WorkspaceTools(str(tmp_path))
        return await AgentLoop(FakeAdapter(), tools.as_tools(), AgentPolicy(max_steps=4)).run(
            [{"role": "user", "content": "create x.txt"}], tools.specs()
        )
    result = asyncio.run(run())
    assert result["status"] == "completed"
    assert (tmp_path / "x.txt").read_text() == "ok"


def test_workspace_blocks_escape(tmp_path):
    tools = WorkspaceTools(str(tmp_path))
    with pytest.raises(ValueError):
        tools.read_file({"path": "../secret.txt"})
