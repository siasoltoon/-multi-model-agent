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


def test_capability_requirement_filters_incompatible_workers():
    registry = WorkerRegistry()
    registry.register(Worker(worker_id="basic", capabilities=["coding", "testing"]))
    registry.register(Worker(worker_id="gpu-worker", capabilities=["coding", "testing", "gpu"], gpu=True))
    orchestrator = WorkerOrchestrator(registry, github_enabled=True)

    selected = orchestrator.select({"capabilities": ["gpu"], "gpu": True})

    assert selected is not None
    assert selected.worker_id == "gpu-worker"
    assert selected.kind == WorkerKind.LAPTOP


def test_resource_requirements_filter_workers():
    registry = WorkerRegistry()
    registry.register(Worker(worker_id="small", cpu_cores=2, memory_mb=4096))
    registry.register(Worker(worker_id="large", cpu_cores=8, memory_mb=16384))
    orchestrator = WorkerOrchestrator(registry, github_enabled=False)

    selected = orchestrator.select({"min_cpu_cores": 4, "min_memory_mb": 8192})

    assert selected is not None
    assert selected.worker_id == "large"


def test_smart_score_can_choose_stronger_worker():
    registry = WorkerRegistry()
    registry.register(Worker(worker_id="laptop-basic", cpu_cores=2, memory_mb=4096))
    registry.register(Worker(worker_id="pc-strong", cpu_cores=12, memory_mb=32768))
    orchestrator = WorkerOrchestrator(registry, github_enabled=False)

    selected = orchestrator.select({"min_cpu_cores": 2, "min_memory_mb": 2048})

    assert selected is not None
    assert selected.worker_id == "pc-strong"


def test_github_remains_fallback_for_generic_requirements():
    registry = WorkerRegistry()
    worker = registry.register(Worker(worker_id="laptop-01"))
    worker.status = WorkerStatus.OFFLINE
    orchestrator = WorkerOrchestrator(registry, github_enabled=True)

    selected = orchestrator.select({"capabilities": ["coding", "ci"]})

    assert selected is not None
    assert selected.kind == WorkerKind.GITHUB


def test_worker_claim_tracks_capacity_and_release_restores_slot():
    registry = WorkerRegistry()
    worker = registry.register(Worker(worker_id="worker-01", max_concurrent_tasks=2))

    registry.claim(worker.worker_id)
    assert worker.active_tasks == 1
    assert worker.available_slots == 1
    assert worker.status == WorkerStatus.BUSY

    registry.claim(worker.worker_id)
    assert worker.active_tasks == 2
    assert worker.available_slots == 0

    try:
        registry.claim(worker.worker_id)
        assert False, "a saturated worker must not accept another task"
    except RuntimeError:
        pass

    registry.release(worker.worker_id)
    assert worker.active_tasks == 1
    assert worker.available_slots == 1
    assert worker.status == WorkerStatus.BUSY

    registry.release(worker.worker_id)
    assert worker.active_tasks == 0
    assert worker.available_slots == 2
    assert worker.status == WorkerStatus.ONLINE


def test_saturated_worker_is_not_a_candidate():
    registry = WorkerRegistry()
    worker = registry.register(Worker(worker_id="worker-01", max_concurrent_tasks=1))
    registry.claim(worker.worker_id)

    orchestrator = WorkerOrchestrator(registry, github_enabled=False)

    assert orchestrator.select() is None


def test_load_aware_scheduler_prefers_worker_with_more_capacity():
    registry = WorkerRegistry()
    busy = registry.register(Worker(worker_id="busy", max_concurrent_tasks=4))
    free = registry.register(Worker(worker_id="free", max_concurrent_tasks=4))
    registry.claim(busy.worker_id)
    registry.claim(busy.worker_id)
    registry.claim(busy.worker_id)

    orchestrator = WorkerOrchestrator(registry, github_enabled=False)
    selected = orchestrator.select()

    assert selected is not None
    assert selected.worker_id == "free"
    assert selected.available_slots == 4


def test_register_resets_previous_runtime_load():
    registry = WorkerRegistry()
    worker = registry.register(Worker(worker_id="worker-01", max_concurrent_tasks=2))
    registry.claim(worker.worker_id)
    assert worker.active_tasks == 1

    replacement = registry.register(Worker(worker_id="worker-01", max_concurrent_tasks=3))

    assert replacement.active_tasks == 0
    assert replacement.available_slots == 3
    assert replacement.status == WorkerStatus.ONLINE
