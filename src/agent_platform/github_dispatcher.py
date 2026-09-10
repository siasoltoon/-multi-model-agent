from __future__ import annotations

import os
import re
from typing import Any
import httpx


class GitHubActionsDispatcher:
    """Dispatch, inspect, and cancel GitHub Actions workers."""

    REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    WORKFLOW_RE = re.compile(r"^[A-Za-z0-9_.@/-]+$")

    def __init__(self, token: str | None = None, api_base: str = "https://api.github.com"):
        self.token = token or os.getenv("AGENT_GITHUB_TOKEN", "")
        self.api_base = api_base.rstrip("/")

    @classmethod
    def _validate_target(cls, repository: str, workflow: str, ref: str) -> None:
        if not cls.REPOSITORY_RE.fullmatch(str(repository or "")):
            raise ValueError("repository must be in owner/name form")
        workflow = str(workflow or "")
        if not workflow or not cls.WORKFLOW_RE.fullmatch(workflow) or ".." in workflow or "//" in workflow:
            raise ValueError("invalid workflow name")
        ref = str(ref or "")
        if not ref or any(ord(ch) < 32 for ch in ref) or any(ch.isspace() for ch in ref):
            raise ValueError("invalid workflow ref")

    def _headers(self) -> dict[str, str]:
        if not self.token:
            raise RuntimeError("AGENT_GITHUB_TOKEN is required for GitHub Actions operations")
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def dispatch(self, repository: str, workflow: str, ref: str, inputs: dict[str, str]) -> dict[str, Any]:
        self._validate_target(repository, workflow, ref)
        response = httpx.post(
            f"{self.api_base}/repos/{repository}/actions/workflows/{workflow}/dispatches",
            headers=self._headers(), json={"ref": ref, "inputs": inputs}, timeout=30,
        )
        response.raise_for_status()
        return {"dispatched": True, "repository": repository, "workflow": workflow, "ref": ref, "inputs": inputs}

    def cancel_run(self, repository: str, run_id: str | int) -> dict[str, Any]:
        """Request cancellation of an active GitHub Actions run; idempotent for finished runs."""
        if not self.REPOSITORY_RE.fullmatch(str(repository or "")):
            raise ValueError("repository must be in owner/name form")
        if not str(run_id).isdigit():
            raise ValueError("run_id must be numeric")
        response = httpx.post(
            f"{self.api_base}/repos/{repository}/actions/runs/{int(run_id)}/cancel",
            headers=self._headers(), timeout=30,
        )
        if response.status_code not in {202, 409}:
            response.raise_for_status()
        return {"cancel_requested": response.status_code == 202, "repository": repository, "run_id": int(run_id), "http_status": response.status_code}
