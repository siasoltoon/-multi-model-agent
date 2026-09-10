from agent_platform.worker_orchestrator import WorkerKind, WorkerOrchestrator
from agent_platform.workers import Worker, WorkerRegistry, WorkerStatus


def test_laptop_is_preferred_when_online():
    registry = WorkerRegistry()
    registry.register(Worker(worker_id="laptop-01", endpoint="http://laptop:8001"))
    orchestrator = WorkerOrchestrator(registry, github_enabled=True)

    selected = orchestrator.select()

    assert selected is not None
    assert selected.kind == WorkerKind.LAPTOP
    assert selected.worker_id == "laptop-01"


def test_github_is_fallback_when_laptop_is_offline():
    registry = WorkerRegistry()
    worker = registry.register(Worker(worker_id="laptop-01"))
    worker.status = WorkerStatus.OFFLINE
    orchestrator = WorkerOrchestrator(registry, github_enabled=True)

    selected = orchestrator.select()

    assert selected is not None
    assert selected.kind == WorkerKind.GITHUB
    assert selected.worker_id == "github-actions"


def test_laptop_is_selected_when_github_is_disabled():
    registry = WorkerRegistry()
    registry.register(Worker(worker_id="laptop-01", endpoint="http://laptop:8001"))
    orchestrator = WorkerOrchestrator(registry, github_enabled=False)

    selected = orchestrator.select()

    assert selected is not None
    assert selected.kind == WorkerKind.LAPTOP
    assert selected.worker_id == "laptop-01"


def test_no_candidate_when_both_are_unavailable():
    registry = WorkerRegistry()
    worker = registry.register(Worker(worker_id="laptop-01"))
    worker.status = WorkerStatus.OFFLINE
    orchestrator = WorkerOrchestrator(registry, github_enabled=False)

    assert orchestrator.select() is None
