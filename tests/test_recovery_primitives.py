from datetime import datetime, timedelta, timezone

from agent_platform.queue import RedisTaskQueue
from agent_platform.workers import Worker, WorkerRegistry, WorkerStatus


class FakePipeline:
    def __init__(self, client):
        self.client = client
        self.ops = []

    def lrem(self, key, count, value):
        self.ops.append(("lrem", key, count, value))
        return self

    def lpush(self, key, value):
        self.ops.append(("lpush", key, value))
        return self

    def execute(self):
        out = []
        for op in self.ops:
            if op[0] == "lrem": out.append(self.client.lrem(*op[1:]))
            else: out.append(self.client.lpush(*op[1:]))
        return out


class FakeRedis:
    def __init__(self):
        self.data = {"agent:tasks": [], "agent:tasks:processing": []}

    def lpush(self, key, value):
        self.data.setdefault(key, []).insert(0, value)
        return len(self.data[key])

    def brpop(self, key, timeout=0):
        if not self.data.get(key): return None
        return key, self.data[key].pop()

    def brpoplpush(self, source, dest, timeout=0):
        if not self.data.get(source): return None
        value = self.data[source].pop()
        self.data.setdefault(dest, []).insert(0, value)
        return value

    def lrem(self, key, count, value):
        values = self.data.get(key, [])
        try: idx = values.index(value)
        except ValueError: return 0
        values.pop(idx)
        return 1

    def lrange(self, key, start, end):
        return self.data.get(key, [])[start:end + 1]

    def pipeline(self): return FakePipeline(self)

    def ping(self): return True


def test_worker_registry_marks_stale_workers_offline():
    registry = WorkerRegistry()
    worker = Worker(worker_id="w1")
    registry.register(worker)
    worker.last_heartbeat = datetime.now(timezone.utc) - timedelta(minutes=5)
    assert registry.mark_stale(30) == ["w1"]
    assert worker.status == WorkerStatus.OFFLINE
    assert registry.online(30) == []


def test_reliable_queue_claim_ack_and_recovery():
    queue = RedisTaskQueue.__new__(RedisTaskQueue)
    queue.client = FakeRedis()
    queue.enqueue("task-a")
    assert queue.claim_reliable(0) == "task-a"
    assert queue.client.data[queue.PROCESSING_KEY] == ["task-a"]
    assert queue.ack("task-a") is True

    queue.enqueue("task-b")
    assert queue.claim_reliable(0) == "task-b"
    assert queue.recover_processing() == ["task-b"]
    assert queue.client.data[queue.PROCESSING_KEY] == []
    assert queue.client.data[queue.READY_KEY] == ["task-b"]
