# Multi-Model Agent

Provider-agnostic, resumable coding-agent control plane with Web Terminal, DAG planning, model routing, bounded self-repair, durable task state, worker leases, optional Redis queue/event bus, and an automatic GitHub Actions worker.

## Runtime flow

`task → durable state → plan → route → inspect/edit → test → diff/review → repair → verified result`

For remote execution: `Web Terminal → Railway control plane → GitHub Actions → persistent branch → callback → checkpoint/resume → PR`.

The model never gets unrestricted host access. Workers own filesystem and command execution, paths are constrained to the workspace, and credentials are resolved from environment/secrets rather than stored in task state.

## Storage and workers

- **SQLite**: zero-config development/single-instance fallback.
- **PostgreSQL**: production durable task/event storage and persistent worker leases.
- **Redis (optional)**: distributed task queue and pub/sub event bus; database remains the source of truth.
- **Workers**: laptop/PC workers and the GitHub Actions worker use the same agent loop.
- **Leases**: PostgreSQL-backed claim/renew/release prevents two workers from owning the same task concurrently.

Install production extras with `pip install -e '.[all]'`.

## Provider/model discovery

The registry can load operator-declared JSON catalogs from `AGENT_PROVIDER_CATALOGS` and discover local Ollama models from `OLLAMA_BASE_URL/api/tags`. It never harvests, guesses, or stores third-party API keys. One model may have multiple endpoints; routing considers context, tool support, reliability, quota state, latency, and task fit.

## Agent loop

The loop supports up to 64 steps, normal 32-step operation, 6 bounded repair attempts, model tool calling, workspace inspection/editing, tests, git diff inspection, timeout checkpointing, and resumable task state. A checkpoint stores the conversation/tool history needed to continue reasoning on the next worker run.

## Web Terminal

Open `/` for the built-in terminal. Creating a task can dispatch it directly to the configured GitHub Actions worker. Task state can be streamed through `/api/tasks/{task_id}/stream`, and durable task events are available through `/api/tasks/{task_id}/events`.

## Automatic GitHub worker

Configure the control plane with `AGENT_GITHUB_TOKEN`, `AGENT_GITHUB_WORKER_REPOSITORY`, `AGENT_GITHUB_WORKER_WORKFLOW`, `AGENT_GITHUB_WORKER_REF`, and `AGENT_PUBLIC_BASE_URL`. The token must be allowed to dispatch workflows and the worker repository must contain `.github/workflows/agent-worker.yml`.

The control plane dispatches a task with a stable task ID and persistent branch name. The worker resumes from that branch and, when a bounded run checkpoints, reports its state after the branch is pushed. The control plane automatically dispatches the next attempt up to the task's `max_worker_attempts` metadata value (default 5). Completed work is kept on the agent branch and the worker opens a PR instead of writing directly to the base branch.

Set `AGENT_GITHUB_CALLBACK_TOKEN` in the control plane and the matching `AGENT_CALLBACK_TOKEN` GitHub Actions secret to authenticate worker callbacks. Set `AGENT_BASE_URL`, `AGENT_MODEL`, and `AGENT_API_KEY` as worker repository secrets for the model endpoint.

## Configuration

Important defaults: 32 normal steps, 64 maximum steps, 6 self-repair attempts, 1800s task timeout, and 300s worker lease. Production deployments should use PostgreSQL and may add Redis for multiple workers.

## Verification

```bash
pytest -q
```

GitHub Actions runs the same test suite on pushes and pull requests. Do not treat a task as verified merely because the model returned a success message; the worker is expected to run relevant tests and inspect the resulting diff.
