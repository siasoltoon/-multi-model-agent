from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .workers import Worker, WorkerRegistry, WorkerStatus


class WorkerKind(str, Enum):
    GITHUB = "github-actions"
    LAPTOP = "laptop"


@dataclass(frozen=True)
class WorkerCandidate:
    kind: WorkerKind
    worker_id: str
    priority: int
    endpoint: str = ""


class WorkerOrchestrator:
    """Selects an execution target from live workers.

    GitHub Actions is preferred for remote execution. A registered laptop/PC
    worker is the automatic fallback. Offline or stale workers are excluded.
    """

    def __init__(self, registry: WorkerRegistry, github_enabled: bool = True):
        self.registry = registry
        self.github_enabled = github_enabled

    def candidates(self) -> list[WorkerCandidate]:
        live = self.registry.online()
        result: list[WorkerCandidate] = []
        if self.github_enabled:
            result.append(WorkerCandidate(WorkerKind.GITHUB, "github-actions", 10))
        for worker in live:
            if worker.status == WorkerStatus.ONLINE:
                result.append(WorkerCandidate(WorkerKind.LAPTOP, worker.worker_id, 20, worker.endpoint))
        return sorted(result, key=lambda item: item.priority)

    def select(self) -> WorkerCandidate | None:
        candidates = self.candidates()
        return candidates[0] if candidates else None

    def select_laptop(self) -> WorkerCandidate | None:
        for candidate in self.candidates():
            if candidate.kind == WorkerKind.LAPTOP:
                return candidate
        return None
