import asyncio

from agent_platform.workspace_tools import WorkspaceTools


def test_safe_environment_filters_secret_markers(monkeypatch):
    monkeypatch.setenv("SAFE_SETTING", "ok")
    monkeypatch.setenv("AGENT_API_KEY", "secret")
    monkeypatch.setenv("DATABASE_PASSWORD", "secret")
    env = WorkspaceTools._safe_environment()
    assert env["SAFE_SETTING"] == "ok"
    assert "AGENT_API_KEY" not in env
    assert "DATABASE_PASSWORD" not in env


def test_run_command_does_not_expose_secret_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_API_KEY", "top-secret")
    tools = WorkspaceTools(str(tmp_path), command_timeout=10)
    result = asyncio.run(tools.run_command({"command": ["python", "-c", "import os; print(os.getenv('AGENT_API_KEY', 'MISSING'))"]}))
    assert result["exit_code"] == 0
    assert "top-secret" not in result["output"]
    assert "MISSING" in result["output"]
