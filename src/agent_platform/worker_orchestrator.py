from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Any

from .workers import Worker, WorkerRegistry, WorkerStatus


class WorkerKind(str, Enum):
    LAPTOP = "laptop"
    GITHUB = "github-actions"


@dataclass(frozen=True)
class WorkerCandidate:
    kind: WorkerKind
    worker_id: str
    priority: int
    endpoint: str = ""
    score: float = 0.0
    matched_capabilities: tuple[str, ...] = ()
    active_tasks: int = 0
    max_concurrent_tasks: int = 1
    available_slots: int = 0


class WorkerOrchestrator:
    """Select a live execution target using capability, resource and load-aware scheduling."""

    _GITHUB_CAPABILITIES = {"coding", "testing", "tools", "github", "linux", "ci"}

    def __init__(self, registry: WorkerRegistry, github_enabled: bool = True):
        self.registry = registry
        self.github_enabled = github_enabled

    @staticmethod
    def _requirements(requirements: Mapping[str, Any] | None) -> tuple[set[str], int | None, int | None, bool, set[str]]:
        requirements = requirements or {}
        capabilities = {str(x).strip().lower() for x in requirements.get("capabilities", []) if str(x).strip()}
        min_cpu = requirements.get("min_cpu_cores")
        min_memory = requirements.get("min_memory_mb")
        gpu = bool(requirements.get("gpu", False))
        models = {str(x).strip().lower() for x in requirements.get("models", []) if str(x).strip()}
        return capabilities, int(min_cpu) if min_cpu is not None else None, int(min_memory) if min_memory is not None else None, gpu, models

    @classmethod
    def _fit(cls, worker: Worker, requirements: Mapping[str, Any] | None) -> tuple[bool, float, tuple[str, ...]]:
        capabilities, min_cpu, min_memory, gpu, models = cls._requirements(requirements)
        worker_capabilities = worker.capability_set()
        matched = tuple(sorted(capabilities & worker_capabilities))
        if not capabilities.issubset(worker_capabilities):
            return False, 0.0, matched
        if min_cpu is not None and (worker.cpu_cores is None or worker.cpu_cores < min_cpu):
            return False, 0.0, matched
        if min_memory is not None and (worker.memory_mb is None or worker.memory_mb < min_memory):
            return False, 0.0, matched
        if gpu and not worker.gpu:
            return False, 0.0, matched
        worker_models = {model.lower() for model in worker.models}
        if models and not models.issubset(worker_models):
            return False, 0.0, matched
        if worker.available_slots <= 0:
            return False, 0.0, matched

        score = 100.0
        score += len(matched) * 10.0
        if min_cpu and worker.cpu_cores:
            score += min(worker.cpu_cores / min_cpu, 4.0) * 5.0
        if min_memory and worker.memory_mb:
            score += min(worker.memory_mb / min_memory, 4.0) * 5.0
        if gpu and worker.gpu:
            score += 15.0
        if models:
            score += len(models) * 5.0
        # Capacity is a bounded load-balancing signal, not a provider preference.
        score += min(worker.available_slots, 4) * 1.0
        score -= worker.load_ratio * 30.0
        return True, score, matched

    def candidates(self, requirements: Mapping[str, Any] | None = None) -> list[WorkerCandidate]:
        live = self.registry.online()
        result: list[WorkerCandidate] = []
        for worker in live:
            if worker.status == WorkerStatus.OFFLINE:
                continue
            fits, score, matched = self._fit(worker, requirements)
            if fits:
                # Real workers are preferred over synthetic GitHub fallback.
                # Among real workers of the same kind, capability/resource/load
                # scoring decides; laptop remains the default execution kind.
                if worker.kind == WorkerKind.LAPTOP.value:
                    priority = 0
                    score += 20.0
                    kind = WorkerKind.LAPTOP
                else:
                    priority = 10
                    kind = WorkerKind.GITHUB
                result.append(WorkerCandidate(
                    kind,
                    worker.worker_id,
                    priority,
                    worker.endpoint,
                    score,
                    matched,
                    worker.active_tasks,
                    worker.max_concurrent_tasks,
                    worker.available_slots,
                ))

        if self.github_enabled:
            github_worker = Worker(
                worker_id="github-actions",
                kind=WorkerKind.GITHUB.value,
                capabilities=sorted(self._GITHUB_CAPABILITIES),
                cpu_cores=None,
                memory_mb=None,
                gpu=False,
                models=[],
                max_concurrent_tasks=256,
            )
            fits, score, matched = self._fit(github_worker, requirements)
            if fits:
                result.append(WorkerCandidate(WorkerKind.GITHUB, "github-actions", 20, "", score, matched, 0, 256, 256))
        return sorted(result, key=lambda item: (-item.score, item.priority, item.worker_id))

    def select(self, requirements: Mapping[str, Any] | None = None) -> WorkerCandidate | None:
        candidates = self.candidates(requirements)
        return candidates[0] if candidates else None

    def select_laptop(self, requirements: Mapping[str, Any] | None = None) -> WorkerCandidate | None:
        for candidate in self.candidates(requirements):
            if candidate.kind == WorkerKind.LAPTOP:
                return candidate
        return None
