import asyncio
from collections.abc import Awaitable, Callable


class TaskEngine:
    def __init__(self) -> None:
        self.tasks: dict[str, dict] = {}

    async def run(self, task_id: str, nodes: list, executor: Callable[[object], Awaitable[None]]) -> None:
        done: set[str] = set()
        pending = {n.id: n for n in nodes}
        while pending:
            ready = [n for n in pending.values() if all(dep in done for dep in n.depends_on)]
            if not ready:
                raise RuntimeError("DAG contains an unresolved dependency or cycle")
            await asyncio.gather(*(executor(n) for n in ready))
            for node in ready:
                done.add(node.id)
                pending.pop(node.id)
            self.tasks[task_id] = {"completed_nodes": sorted(done), "total_nodes": len(nodes)}
