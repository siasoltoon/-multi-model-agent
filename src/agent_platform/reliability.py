from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any


_SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|token|secret|password|authorization)\s*[:=]\s*([^\s,;]+)"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+"),
)


def redact_secrets(value: Any) -> Any:
    if isinstance(value, str):
        text = value
        for pattern in _SECRET_PATTERNS:
            text = pattern.sub(lambda m: m.group(1) + "=<redacted>" if m.lastindex and m.lastindex >= 2 else "Bearer <redacted>", text)
        return text
    if isinstance(value, dict):
        return {str(k): ("<redacted>" if any(x in str(k).lower() for x in ("key", "token", "secret", "password")) else redact_secrets(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_secrets(v) for v in value]
    return value


def idempotency_key(task_id: str, phase: str, attempt: int = 0) -> str:
    raw = f"{task_id}:{phase}:{attempt}".encode()
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class RetryPolicy:
    attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0

    def delay(self, attempt: int) -> float:
        return min(self.max_delay, self.base_delay * (2 ** max(0, attempt - 1)))


def checkpoint_payload(task_id: str, phase: str, messages: list[dict[str, Any]], *, steps: int, repairs: int) -> dict[str, Any]:
    return {
        "version": 1,
        "task_id": task_id,
        "phase": phase,
        "steps": steps,
        "repairs": repairs,
        "messages": redact_secrets(messages),
    }
