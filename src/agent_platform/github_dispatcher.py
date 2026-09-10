from __future__ import annotations

import os
from typing import Any
import httpx


class GitHubActionsDispatcher:
    """Dispatch, inspect, and cancel GitHub Actions workers."""

    def __init__(self, token: str | None = None, api_base: str = "https://api.github.com"):
        self.token = token or os.getenv("AGENT_GITHUB_TOKEN", "")
        self.api_base = api_base.rstrip("/")

    def _headers(self) -> dict[str, str]:
        if not self.token:
            raise RuntimeError("AGENT_GITHUB_TOKEN is required for GitHub Actions operations")
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def dispatch(self, repository: str, workflow: str, ref: str, inputs: dict[str, str]) -> dict[str, Any]:
        response = httpx.post(
            f"{self.api_base}/repos/{repository}/actions/workflows/{workflow}/dispatches",
            headers=self._headers(), json={"ref": ref, "inputs": inputs}, timeout=30,
        )
        response.raise_for_status()
        return {"dispatched": True, "repository": repository, "workflow": workflow, "ref": ref, "inputs": inputs}

    def cancel_run(self, repository: str, run_id: str | int) -> dict[str, Any]:
        """Request cancellation of an active GitHub Actions run; idempotent for finished runs."""
        if not str(run_id).isdigit():
            raise ValueError("run_id must be numeric")
        response = httpx.post(
            f"{self.api_base}/repos/{repository}/actions/runs/{int(run_id)}/cancel",
            headers=self._headers(), timeout=30,
        )
        if response.status_code not in {202, 409}:
            response.raise_for_status()
        return {"cancel_requested": response.status_code == 202, "repository": repository, "run_id": int(run_id), "http_status": response.status_code}
