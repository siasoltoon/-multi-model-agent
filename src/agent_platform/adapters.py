from dataclasses import dataclass
from typing import Any, Protocol

import httpx


@dataclass
class ModelResponse:
    text: str
    raw: dict[str, Any]
    usage: dict[str, Any]


class ModelAdapter(Protocol):
    async def generate(self, messages: list[dict[str, str]], *, tools: list[dict[str, Any]] | None = None) -> ModelResponse: ...


class OpenAICompatibleAdapter:
    """Works with OpenAI-compatible endpoints; provider-specific adapters can subclass it."""
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 180):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    async def generate(self, messages, *, tools=None) -> ModelResponse:
        payload = {"model": self.model, "messages": messages}
        if tools:
            payload["tools"] = tools
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.url, headers={"Authorization": f"Bearer {self.api_key}"}, json=payload)
            response.raise_for_status()
            data = response.json()
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        return ModelResponse(message.get("content", ""), data, data.get("usage", {}))
