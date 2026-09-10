from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from .reliability import RetryPolicy, redact_secrets


@dataclass
class ModelResponse:
    text: str
    raw: dict[str, Any]
    usage: dict[str, Any]
    tool_calls: list[dict[str, Any]]


class ModelAdapter(Protocol):
    async def generate(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None) -> ModelResponse: ...


class ModelRequestError(RuntimeError):
    def __init__(self, status_code: int | None, message: str, retryable: bool = False, retry_after: float | None = None):
        self.status_code = status_code
        self.retryable = retryable
        self.retry_after = retry_after
        super().__init__(redact_secrets(message)[:1000])


class OpenAICompatibleAdapter:
    """OpenAI-compatible chat-completions adapter with bounded retries and tool-call support."""
    def __init__(self, base_url: str, api_key: str = "", model: str = "", timeout: float = 180, retry_policy: RetryPolicy | None = None):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.retry_policy = retry_policy or RetryPolicy()

    async def _request(self, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.retry_policy.attempts + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(self.url, headers=headers, json=payload)
                if response.is_success:
                    data = response.json()
                    if not isinstance(data, dict):
                        raise ModelRequestError(response.status_code, "provider returned a non-object response")
                    return data
                retryable = response.status_code == 429 or response.status_code >= 500
                retry_after = None
                raw_retry = response.headers.get("retry-after")
                if raw_retry:
                    try:
                        retry_after = max(0.0, float(raw_retry))
                    except ValueError:
                        pass
                try:
                    error_payload = response.json()
                    message = error_payload.get("error", {}).get("message", "provider request failed") if isinstance(error_payload, dict) else "provider request failed"
                except (ValueError, TypeError):
                    message = response.text[:500] or "provider request failed"
                raise ModelRequestError(response.status_code, str(message), retryable, retry_after)
            except ModelRequestError as exc:
                last_error = exc
                if not exc.retryable or attempt >= self.retry_policy.attempts:
                    raise
                await asyncio.sleep(exc.retry_after if exc.retry_after is not None else self.retry_policy.delay(attempt))
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = ModelRequestError(None, str(exc), True)
                if attempt >= self.retry_policy.attempts:
                    raise last_error
                await asyncio.sleep(self.retry_policy.delay(attempt))
        raise last_error or RuntimeError("model request failed")

    async def generate(self, messages, *, tools=None) -> ModelResponse:
        payload = {"model": self.model, "messages": messages}
        if tools:
            payload["tools"] = tools
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = await self._request(payload, headers)
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        calls: list[dict[str, Any]] = []
        for call in message.get("tool_calls", []) or []:
            fn = call.get("function", {})
            calls.append({"id": call.get("id"), "name": fn.get("name", ""), "arguments": fn.get("arguments", "{}")})
        return ModelResponse(message.get("content") or "", data, data.get("usage", {}), calls)


class FailoverAdapter:
    """Try ranked provider adapters once per request and disable failed endpoints for this task."""

    def __init__(self, adapters: list[tuple[str, ModelAdapter]], on_failure=None):
        if not adapters:
            raise ValueError("at least one adapter is required")
        self.adapters = adapters
        self.on_failure = on_failure
        self.active_endpoint_id = adapters[0][0]
        self.disabled: set[str] = set()

    async def generate(self, messages, *, tools=None) -> ModelResponse:
        last_error: Exception | None = None
        for endpoint_id, adapter in self.adapters:
            if endpoint_id in self.disabled:
                continue
            try:
                response = await adapter.generate(messages, tools=tools)
                self.active_endpoint_id = endpoint_id
                return response
            except Exception as exc:
                last_error = exc
                self.disabled.add(endpoint_id)
                if self.on_failure:
                    self.on_failure(endpoint_id, exc)
        raise last_error or RuntimeError("all model endpoints failed")
