from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .workers import WorkerRegistry, WorkerStatus


class WorkerKind(str, Enum):
    LAPTOP = "laptop"
    GITHUB = "github-actions"


@dataclass(frozen=True)
class WorkerCandidate:
    kind: WorkerKind
    worker_id: str
    priority: int
    endpoint: str = ""


class WorkerOrchestrator:
    """Select a live execution target using laptop-first failover."""

    def __init__(self, registry: WorkerRegistry, github_enabled: bool = True):
        self.registry = registry
        self.github_enabled = github_enabled

    def candidates(self) -> list[WorkerCandidate]:
        live = self.registry.online()
        result: list[WorkerCandidate] = []
        for worker in live:
            if worker.status == WorkerStatus.ONLINE:
                result.append(WorkerCandidate(WorkerKind.LAPTOP, worker.worker_id, 10, worker.endpoint))
        if self.github_enabled:
            result.append(WorkerCandidate(WorkerKind.GITHUB, "github-actions", 20))
        return sorted(result, key=lambda item: (item.priority, item.worker_id))

    def select(self) -> WorkerCandidate | None:
        candidates = self.candidates()
        return candidates[0] if candidates else None

    def select_laptop(self) -> WorkerCandidate | None:
        for candidate in self.candidates():
            if candidate.kind == WorkerKind.LAPTOP:
                return candidate
        return None
