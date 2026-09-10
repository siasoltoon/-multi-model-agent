from __future__ import annotations

import hashlib
import hmac
import re
import time
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


def callback_signature(secret: str, timestamp: str, body: bytes) -> str:
    """Return a GitHub-webhook-style HMAC signature for callback bodies."""
    digest = hmac.new(secret.encode(), timestamp.encode() + b"." + body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def verify_callback_signature(secret: str, timestamp: str, body: bytes, signature: str, *, tolerance_seconds: int = 300, now: float | None = None) -> bool:
    if not secret or not timestamp or not signature:
        return False
    try:
        ts = int(timestamp)
    except (TypeError, ValueError):
        return False
    current = time.time() if now is None else now
    if abs(current - ts) > max(1, tolerance_seconds):
        return False
    expected = callback_signature(secret, timestamp, body)
    return hmac.compare_digest(expected, signature)
