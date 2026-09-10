from __future__ import annotations

import os
from typing import Any
import httpx


class GitHubActionsDispatcher:
    """Dispatch and authenticate GitHub Actions workers using an explicit token."""

    def __init__(self, token: str | None = None, api_base: str = "https://api.github.com"):
        self.token = token or os.getenv("AGENT_GITHUB_TOKEN", "")
        self.api_base = api_base.rstrip("/")

    def dispatch(self, repository: str, workflow: str, ref: str, inputs: dict[str, str]) -> dict[str, Any]:
        if not self.token:
            raise RuntimeError("AGENT_GITHUB_TOKEN is required for GitHub Actions dispatch")
        url = f"{self.api_base}/repos/{repository}/actions/workflows/{workflow}/dispatches"
        headers = {"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        response = httpx.post(url, headers=headers, json={"ref": ref, "inputs": inputs}, timeout=30)
        response.raise_for_status()
        return {"dispatched": True, "repository": repository, "workflow": workflow, "ref": ref, "inputs": inputs}
