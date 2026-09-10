# Multi-Model Agent

Provider-agnostic, resumable coding-agent control plane with terminal-first execution, DAG planning, adaptive model routing, bounded self-repair, durable task state, worker leases, optional Redis queue/event bus, and an automatic GitHub Actions worker.

## Runtime flow

`task → durable state → plan → discover models → route → inspect/edit → test → diff/review → repair → verified result`

For remote execution: `terminal → Railway control plane → GitHub Actions → persistent branch → callback → checkpoint/resume → PR`.

The project does not require a dashboard/Web Terminal UI. The CLI is the primary operator interface.

The model never gets unrestricted host access. Workers own filesystem and command execution, paths are constrained to the workspace, and credentials are resolved from environment/secrets rather than stored in task state.

## Storage and workers

- **SQLite**: zero-config development/single-instance fallback.
- **PostgreSQL**: production durable task/event storage and persistent worker leases.
- **Redis (optional)**: distributed task queue and pub/sub event bus; database remains the source of truth.
- **Workers**: laptop/PC workers and the GitHub Actions worker use the same agent loop.
- **Leases**: PostgreSQL-backed claim/renew/release prevents two workers from owning the same task concurrently.

Install production extras with `pip install -e '.[all]'`.

## Provider/model discovery and routing

The control plane automatically discovers models from configured provider catalogs when their credentials exist. Built-in OpenAI-compatible catalogs currently cover **OpenRouter, Groq, Cerebras, Together AI, Fireworks AI, and Mistral**, plus local **Ollama** discovery and operator-declared JSON catalogs through `AGENT_PROVIDER_CATALOGS`.

For every discovered model the registry records provider, model ID, context window, tool capability, billing classification, task-fit metadata and the secret environment-variable name. Secrets themselves are never written to task state or the model catalog.

At runtime, discovered endpoints are registered with the Smart Router. Routing is free/local-first, then considers task fit, reliability, quota, latency, context and tool support. If a provider fails, the agent can fail over to the next ranked endpoint without requiring the user to manually select another model. Failed endpoints are disabled for the current task and put into the router's health/cooldown state for later tasks.

**API keys are not automatically created or obtained.** The operator must provide provider credentials through environment variables or Railway/GitHub Actions secrets. One provider key normally unlocks many models from that provider; you do not need a separate key for every model.

Current built-in provider variables:

- `OPENROUTER_API_KEY`
- `GROQ_API_KEY`
- `CEREBRAS_API_KEY`
- `TOGETHER_API_KEY`
- `FIREWORKS_API_KEY`
- `MISTRAL_API_KEY`

Optional runtime controls:

- `AGENT_MAX_PROVIDER_FAILOVERS=3`
- `AGENT_MODEL_REQUEST_TIMEOUT=180`
- `OLLAMA_BASE_URL=http://127.0.0.1:11434`
- `AGENT_PROVIDER_CATALOGS=`

The `/api/providers` endpoint exposes the discovered model inventory, health, billing classification, context and routing telemetry without exposing API keys. `POST /api/providers/discover` refreshes the catalog and `POST /api/providers/health` probes registered endpoints.

## Token/context saver

Long coding tasks can accumulate very large tool outputs. Before every model request, the context saver compacts only historical tool output while preserving system/task instructions and the most recent active turns verbatim. It also de-duplicates repeated tool results and keeps high-signal lines such as errors, test failures, paths, warnings and command summaries.

The default target is **85% reduction of historical tool-output characters**, configurable between 80% and 90%. This is a context/payload reduction target, not a guarantee that every provider bill will drop by exactly 85%: provider tokenization, system tokens and recent turns still contribute to usage. The design deliberately favors retaining information needed for correctness over blindly truncating the conversation.

Environment controls:

- `AGENT_CONTEXT_TARGET_REDUCTION=0.85`
- `AGENT_CONTEXT_KEEP_RECENT=8`
- `AGENT_CONTEXT_MAX_TOOL_CHARS=6000`

## Agent loop

The loop supports up to 64 steps, normal 32-step operation, 6 bounded repair attempts, model tool calling, workspace inspection/editing, tests, git diff inspection, timeout checkpointing, context compaction and resumable task state. A checkpoint stores the conversation/tool history needed to continue reasoning on the next worker run.

## Terminal CLI

```bash
multi-model-agent submit "ساخت یک ربات تلگرام دانلودر"
multi-model-agent run-local "این پروژه را بررسی کن و باگ‌های آن را اصلاح کن"
multi-model-agent status TASK_ID
multi-model-agent watch TASK_ID
multi-model-agent plan TASK_ID
multi-model-agent events TASK_ID
multi-model-agent resume TASK_ID
multi-model-agent cancel TASK_ID
```

Creating a remote task can dispatch it directly to the configured GitHub Actions worker. Task state can be streamed through `/api/tasks/{task_id}/stream`, and durable task events are available through `/api/tasks/{task_id}/events`.

## Automatic GitHub worker

Configure the control plane with `AGENT_GITHUB_TOKEN`, `AGENT_GITHUB_WORKER_REPOSITORY`, `AGENT_GITHUB_WORKER_WORKFLOW`, `AGENT_GITHUB_WORKER_REF`, and `AGENT_PUBLIC_BASE_URL`. The token must be allowed to dispatch workflows and the worker repository must contain `.github/workflows/agent-worker.yml`.

The control plane dispatches a task with a stable task ID and persistent branch name. The worker resumes from that branch and, when a bounded run checkpoints, reports its state after the branch is pushed. The control plane automatically dispatches the next attempt up to the task's `max_worker_attempts` metadata value (default 5). Completed work is kept on the agent branch and the worker opens a PR instead of writing directly to the base branch.

Set `AGENT_GITHUB_CALLBACK_TOKEN` in the control plane and the matching `AGENT_CALLBACK_TOKEN` GitHub Actions secret to authenticate worker callbacks. Provider API keys can be supplied to the worker as repository secrets using the same variables listed above; no API key is committed to the repository.

## Configuration

Important defaults: 32 normal steps, 64 maximum steps, 6 self-repair attempts, 1800s task timeout, 180s model request timeout, 3 provider failover candidates, and 300s worker lease. Production deployments should use PostgreSQL and may add Redis for multiple workers.

## Verification

```bash
pytest -q
```

GitHub Actions runs the same test suite on pushes and pull requests. Do not treat a task as verified merely because the model returned a success message; the worker is expected to run relevant tests and inspect the resulting diff.