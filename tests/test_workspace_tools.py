import asyncio

import pytest

from agent_platform.workspace_tools import WorkspaceTools


def test_run_command_allows_safe_inspection_commands(tmp_path):
    tools = WorkspaceTools(str(tmp_path))
    assert "run_command" in tools.as_tools()
    assert "pwd" in tools.ALLOWED_COMMANDS
    assert "ls" in tools.ALLOWED_COMMANDS
    assert "find" in tools.ALLOWED_COMMANDS
    assert "cat" in tools.ALLOWED_COMMANDS


def test_run_command_supports_safe_and_chain(tmp_path):
    tools = WorkspaceTools(str(tmp_path))
    result = asyncio.run(tools.run_command({"command": "pwd && git status --short"}))
    assert result["exit_code"] == 0


def test_run_command_rejects_shell_injection(tmp_path):
    tools = WorkspaceTools(str(tmp_path))
    with pytest.raises(ValueError, match="shell operator"):
        asyncio.run(tools.run_command({"command": "pwd; touch escaped.txt"}))
    with pytest.raises(ValueError, match="shell operator"):
        asyncio.run(tools.run_command({"command": "pwd | cat"}))
    with pytest.raises(ValueError, match="shell operator"):
        asyncio.run(tools.run_command({"command": "pwd && $(touch escaped.txt)"}))


def test_run_command_rejects_unallowlisted_commands(tmp_path):
    tools = WorkspaceTools(str(tmp_path))
    with pytest.raises(ValueError, match="not allowlisted"):
        asyncio.run(tools.run_command({"command": "curl https://example.com"}))


def test_run_command_keeps_workspace_path_boundary(tmp_path):
    tools = WorkspaceTools(str(tmp_path))
    with pytest.raises(ValueError, match="path escapes workspace"):
        tools.read_file({"path": "../outside.txt"})


def test_run_command_blocks_package_manager_global_options(tmp_path):
    tools = WorkspaceTools(str(tmp_path))
    with pytest.raises(ValueError, match="blocked package-management"):
        asyncio.run(tools.run_command({"command": "pip install --user example"}))
