# Multi-Model Agent

A provider-agnostic coding-agent control plane with a Web Terminal, DAG planning, smart model routing, durable task state, bounded self-repair, sandboxed worker tools, and a GitHub Actions worker.

## Runtime flow

`task → plan → route model → inspect/edit → run tests → inspect diff → repair → verified result`

The agent is bounded by configurable step, timeout, and repair limits. It does not claim completion merely because a model returned text: worker-side tools are used to inspect and verify the workspace.

## Provider/model discovery

The registry can load operator-declared JSON catalogs from `AGENT_PROVIDER_CATALOGS` and discover local Ollama models from `OLLAMA_BASE_URL/api/tags`. Credentials are resolved only from environment variables; the project never harvests or stores third-party API keys.

Each endpoint is scored using task fit, reliability, context capacity, tool support, latency, health, and quota state. Provider catalogs should contain only metadata the operator is authorized to use.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -e '.[dev]'
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open `/` for the Web Terminal. Create a task with `POST /api/tasks`, then execute it on a worker with `POST /api/tasks/{task_id}/run?workspace=/path/to/repository`.

## GitHub Actions worker

`.github/workflows/agent-worker.yml` is manually dispatchable. It accepts a prompt, step budget, and target repository, checks the repository out, runs the same bounded agent loop, and pushes only the resulting workspace changes.

Configure repository secrets when using a remote model:

- `AGENT_BASE_URL`
- `AGENT_MODEL`
- `AGENT_API_KEY`
- `AGENT_GITHUB_TOKEN` when the target repository is different from the control-plane repository or requires broader permissions

The worker is intentionally conservative: workspace paths are constrained, shell commands are allowlisted, and model credentials are never written to task state.

## Configuration

Important defaults are `32` normal steps, `64` maximum steps, `6` self-repair attempts, and an `1800s` execution timeout. SQLite is the development persistence backend; the storage and worker interfaces are designed so a PostgreSQL/queue backend can be added without changing the agent loop.

## Verification

Run:

```bash
pytest -q
```

The GitHub Actions test workflow runs the same test suite on pushes and pull requests.
