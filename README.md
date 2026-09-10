# Multi-Model Agent

A production-oriented, provider-agnostic coding agent control plane with a Web Terminal, DAG orchestration, smart model routing, resumable tasks, and ephemeral workers.

## Goals
- Web Terminal as the primary UI
- Analyze → plan → execute → test → review → self-repair
- Multi-model/provider routing with health and quota awareness
- DAG-based orchestration instead of a fixed linear pipeline
- Persistent task state and checkpoints
- Laptop workers and GitHub Actions workers
- GitHub-first repository operations

## Quick start

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -e .[dev]
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/terminal`.

This first implementation is intentionally dependency-light. Provider adapters, PostgreSQL/Redis persistence, GitHub Actions dispatch, and additional agent roles can be added behind the stable interfaces in `app/`.
