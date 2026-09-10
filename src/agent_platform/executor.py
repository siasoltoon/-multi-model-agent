from dataclasses import dataclass
from typing import Callable, Awaitable, Any

from .dag import DagNode, NodeStatus, TaskDag
from .models import Task, TaskStatus


@dataclass
class ExecutionPolicy:
    max_steps: int = 64
    repair_attempts: int = 6
    checkpoint_every: int = 1


class DagExecutor:
    """Execute a DAG with bounded retries and checkpoint callbacks."""
    def __init__(self, policy: ExecutionPolicy | None = None):
        self.policy = policy or ExecutionPolicy()

    async def run(self, task: Task, dag: TaskDag, execute: Callable[[DagNode], Any], checkpoint: Callable[[Task, DagNode], None] | None = None):
        task.status = TaskStatus.RUNNING
        while not dag.is_complete():
            ready = dag.ready()
            if not ready:
                failed = [n for n in dag.nodes.values() if n.status == NodeStatus.FAILED]
                if failed:
                    task.status, task.error = TaskStatus.FAILED, failed[0].error
                    return task
                raise RuntimeError("DAG is blocked")
            for node in ready:
                if task.current_step >= task.max_steps:
                    task.status = TaskStatus.CHECKPOINTED
                    if checkpoint:
                        checkpoint(task, node)
                    return task
                node.status = NodeStatus.RUNNING
                node.attempts += 1
                task.current_step += 1
                try:
                    value = execute(node)
                    node.result = await value if hasattr(value, "__await__") else value
                    dag.mark_completed(node.id, node.result)
                except Exception as exc:
                    if node.attempts <= self.policy.repair_attempts:
                        node.status = NodeStatus.PENDING
                        node.error = str(exc)
                        task.repair_attempts += 1
                        continue
                    dag.mark_failed(node.id, str(exc))
                    task.status, task.error = TaskStatus.FAILED, str(exc)
                    return task
                if checkpoint and task.current_step % self.policy.checkpoint_every == 0:
                    checkpoint(task, node)
        task.status = TaskStatus.COMPLETED
        task.result = {"nodes": {k: v.result for k, v in dag.nodes.items()}}
        return task
