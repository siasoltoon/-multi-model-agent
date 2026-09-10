from dataclasses import dataclass
from typing import Any, Protocol

import httpx


@dataclass
class ModelResponse:
    text: str
    raw: dict[str, Any]
    usage: dict[str, Any]
    tool_calls: list[dict[str, Any]]


class ModelAdapter(Protocol):
    async def generate(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None) -> ModelResponse: ...


class OpenAICompatibleAdapter:
    """OpenAI-compatible chat-completions adapter with native tool-call support."""
    def __init__(self, base_url: str, api_key: str = "", model: str = "", timeout: float = 180):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    async def generate(self, messages, *, tools=None) -> ModelResponse:
        payload = {"model": self.model, "messages": messages}
        if tools:
            payload["tools"] = tools
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        calls: list[dict[str, Any]] = []
        for call in message.get("tool_calls", []) or []:
            fn = call.get("function", {})
            calls.append({"id": call.get("id"), "name": fn.get("name", ""), "arguments": fn.get("arguments", "{}")})
        return ModelResponse(message.get("content") or "", data, data.get("usage", {}), calls)


class FailoverAdapter:
    """Try ranked provider adapters in order and remember the active endpoint."""

    def __init__(self, adapters: list[tuple[str, ModelAdapter]], on_failure=None):
        if not adapters:
            raise ValueError("at least one adapter is required")
        self.adapters = adapters
        self.on_failure = on_failure
        self.active_endpoint_id = adapters[0][0]

    async def generate(self, messages, *, tools=None) -> ModelResponse:
        last_error: Exception | None = None
        for endpoint_id, adapter in self.adapters:
            try:
                response = await adapter.generate(messages, tools=tools)
                self.active_endpoint_id = endpoint_id
                return response
            except Exception as exc:
                last_error = exc
                if self.on_failure:
                    self.on_failure(endpoint_id, exc)
        raise last_error or RuntimeError("all model endpoints failed")
