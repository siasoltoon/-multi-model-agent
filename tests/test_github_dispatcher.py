import httpx
import pytest

from agent_platform.github_dispatcher import GitHubActionsDispatcher


def test_cancel_run_posts_to_github(monkeypatch):
    calls = {}

    def fake_post(url, **kwargs):
        calls.update(url=url, kwargs=kwargs)
        return httpx.Response(202, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    result = GitHubActionsDispatcher("token").cancel_run("owner/repo", "123")

    assert result["cancel_requested"] is True
    assert result["run_id"] == 123
    assert calls["url"].endswith("/repos/owner/repo/actions/runs/123/cancel")
    assert calls["kwargs"]["headers"]["Authorization"] == "Bearer token"


def test_cancel_run_rejects_non_numeric_run_id():
    with pytest.raises(ValueError, match="numeric"):
        GitHubActionsDispatcher("token").cancel_run("owner/repo", "run-123")
