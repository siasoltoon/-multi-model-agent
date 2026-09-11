from __future__ import annotations

import argparse
import asyncio
import os
import socket
from contextlib import suppress
from pathlib import Path

import httpx

from .discovery import ProviderDiscovery
from .models import Task
from .router import ModelEndpoint, SmartRouter
from .runner import run_task


def _base_url() -> str:
    return os.getenv("AGENT_CONTROL_PLANE_URL", "http://127.0.0.1:8000").rstrip("/")


def _headers() -> dict[str, str]:
    token = os.getenv("AGENT_WORKER_AUTH_TOKEN", "")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _worker_id() -> str:
    return os.getenv("AGENT_WORKER_ID", f"laptop-{socket.gethostname().lower()}")


def _workspace() -> str:
    return str(Path(os.getenv("AGENT_WORKSPACE", os.getcwd())).resolve())


async def _router() -> SmartRouter:
    router = SmartRouter()
    for item in await ProviderDiscovery().discover():
        router.register(ModelEndpoint(
            id=f"{item.provider}:{item.model}:{item.base_url}",
            provider=item.provider,
            model=item.model,
            base_url=item.base_url,
            context_window=item.context_window,
            tool_support=item.tool_support,
            task_fit=item.task_fit,
            reliability=item.reliability,
            latency_ms=item.latency_ms,
            api_key_env=item.api_key_env,
            metadata=item.metadata,
        ))
    if not router.endpoints:
        base = os.getenv("AGENT_BASE_URL", "")
        model = os.getenv("AGENT_MODEL", "")
        if base and model:
            router.register(ModelEndpoint("env", "env", model, base_url=base, tool_support=True))
    return router


async def _lease_heartbeat(client: httpx.AsyncClient, base_url: str, worker_id: str, lease_id: str, headers: dict[str, str], interval: float) -> None:
    while True:
        await asyncio.sleep(interval)
        try:
            response = await client.post(
                f"{base_url}/api/workers/{worker_id}/heartbeat",
                headers=headers,
                params={"status": "busy", "lease_id": lease_id},
            )
            response.raise_for_status()
        except asyncio.CancelledError:
            raise
        except httpx.HTTPError as exc:
            print(f"worker heartbeat error: {exc}", flush=True)


async def run_worker() -> int:
    worker_id = _worker_id()
    base_url = _base_url()
    workspace = _workspace()
    poll_seconds = float(os.getenv("AGENT_WORKER_POLL_SECONDS", "5"))
    heartbeat_seconds = float(os.getenv("AGENT_WORKER_HEARTBEAT_SECONDS", "15"))
    endpoint = os.getenv("AGENT_WORKER_ENDPOINT", "")
    headers = _headers()

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{base_url}/api/workers/register",
            headers=headers,
            json={"worker_id": worker_id, "endpoint": endpoint, "status": "online"},
        )
        response.raise_for_status()
        router = await _router()
        if not router.endpoints:
            raise RuntimeError("No model endpoint discovered for laptop worker")

        while True:
            try:
                heartbeat = await client.post(f"{base_url}/api/workers/{worker_id}/heartbeat", headers=headers, params={"status": "online"})
                heartbeat.raise_for_status()
                response = await client.post(f"{base_url}/api/tasks/claim-next", headers=headers)
                if response.status_code == 204:
                    await asyncio.sleep(poll_seconds)
                    continue
                response.raise_for_status()
                assignment = response.json()
                task = Task.model_validate(assignment["task"])
                lease_id = str(assignment["lease_id"])
                attempt = int(assignment.get("attempt", task.attempts))

                heartbeat = await client.post(f"{base_url}/api/workers/{worker_id}/heartbeat", headers=headers, params={"status": "busy", "lease_id": lease_id})
                heartbeat.raise_for_status()
                lease_heartbeat = asyncio.create_task(_lease_heartbeat(client, base_url, worker_id, lease_id, headers, max(5.0, heartbeat_seconds)))
                try:
                    result = await run_task(task, router, workspace)
                    payload = dict(result)
                    payload.update({
                        "worker_id": worker_id,
                        "attempt": attempt,
                        "steps": task.current_step,
                        "repairs": task.repair_attempts,
                        "active_role": payload.get("active_role"),
                    })
                except Exception as exc:
                    payload = {"status": "failed", "error": str(exc), "worker_id": worker_id, "attempt": attempt, "steps": task.current_step, "repairs": task.repair_attempts}
                finally:
                    lease_heartbeat.cancel()
                    with suppress(asyncio.CancelledError):
                        await lease_heartbeat

                heartbeat = await client.post(f"{base_url}/api/workers/{worker_id}/heartbeat", headers=headers, params={"status": "online", "lease_id": lease_id})
                heartbeat.raise_for_status()
                callback = await client.post(f"{base_url}/api/tasks/{task.id}/worker-callback", headers=headers, json=payload)
                callback.raise_for_status()
                await client.delete(f"{base_url}/api/workers/{worker_id}/lease/{lease_id}", headers=headers)
            except (httpx.HTTPError, RuntimeError) as exc:
                print(f"worker error: {exc}", flush=True)
                await asyncio.sleep(poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-Model Agent laptop worker")
    parser.add_argument("--workspace", default=None)
    args = parser.parse_args()
    if args.workspace:
        os.environ["AGENT_WORKSPACE"] = args.workspace
    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
