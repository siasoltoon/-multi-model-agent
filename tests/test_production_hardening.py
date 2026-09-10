from __future__ import annotations

import pytest

from agent_platform.app import _validate_callback_url, _validate_github_target


def test_github_target_validation_accepts_normal_target():
    _validate_github_target("owner/repo", "agent-worker.yml", "main", "agent/task-123")


@pytest.mark.parametrize(
    "repository,workflow,ref,branch",
    [
        ("owner", "agent-worker.yml", "main", "agent/task-1"),
        ("owner/repo/extra", "agent-worker.yml", "main", "agent/task-1"),
        ("owner/repo", "../agent-worker.yml", "main", "agent/task-1"),
        ("owner/repo", "agent-worker.yml", "../main", "agent/task-1"),
        ("owner/repo", "agent-worker.yml", "main", "../agent/task-1"),
    ],
)
def test_github_target_validation_rejects_unsafe_values(repository, workflow, ref, branch):
    with pytest.raises(Exception):
        _validate_github_target(repository, workflow, ref, branch)


def test_callback_url_validation_accepts_https():
    _validate_callback_url("https://example.test/api/tasks/123/worker-callback")


@pytest.mark.parametrize("url", ["/callback", "ftp://example.test/callback", "https://user:pass@example.test/callback"])
def test_callback_url_validation_rejects_unsafe_urls(url):
    with pytest.raises(Exception):
        _validate_callback_url(url)
