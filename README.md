# Multi-Model Agent

Provider-agnostic, resumable coding-agent control plane with Web Terminal, DAG planning, model routing, bounded self-repair, durable task state, worker leases, optional Redis queue/event bus, and a GitHub Actions worker.

## Runtime flow

`task → durable state → plan → route → inspect/edit → test → diff/review → repair → verified result`

The model never gets unrestricted host access. Workers own filesystem and command execution, paths are constrained to the workspace, and credentials are resolved from environment/secrets rather than stored in task state.

## Storage and workers

- **SQLite**: zero-config development/single-instance fallback.
- **PostgreSQL**: production durable task/event storage and persistent worker leases.
- **Redis (optional)**: distributed task queue and pub/sub event bus; database remains the source of truth.
- **Workers**: laptop/PC workers and the GitHub Actions worker can use the same agent loop.
- **Leases**: PostgreSQL-backed claim/renew/release prevents two workers from owning the same task concurrently.

Install production extras with `pip install -e '.[all]'`.

## Provider/model discovery

The registry can load operator-declared JSON catalogs from `AGENT_PROVIDER_CATALOGS` and discover local Ollama models from `OLLAMA_BASE_URL/api/tags`. It never harvests, guesses, or stores third-party API keys. One model may have multiple endpoints; routing considers context, tool support, reliability, quota state, latency, and task fit.

## Agent loop

The loop supports up to 64 steps, normal 32-step operation, 6 bounded repair attempts, model tool calling, workspace inspection/editing, tests, git diff inspection, timeout checkpointing, and resumable task state.

## Web Terminal

Open `/` for the built-in terminal. Task state can be streamed through `/api/tasks/{task_id}/stream`, and durable task events are available through `/api/tasks/{task_id}/events`.

## GitHub Actions worker

`.github/workflows/agent-worker.yml` is manually dispatchable. It checks out the requested repository, runs the bounded worker loop, and pushes verified workspace changes. Configure `AGENT_BASE_URL`, `AGENT_MODEL`, and `AGENT_API_KEY` as repository secrets when a remote model is used. Use `AGENT_WORKER_AUTH_TOKEN` for control-plane worker APIs.

## Configuration

Important defaults: 32 normal steps, 64 maximum steps, 6 self-repair attempts, 1800s task timeout, and 300s worker lease. Production deployments should use PostgreSQL and may add Redis for multiple workers.

## Verification

```bash
pytest -q
```

GitHub Actions runs the same test suite on pushes and pull requests.
