from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .adapters import ModelAdapter, ModelResponse


@dataclass
class AgentPolicy:
    max_steps: int = 64
    repair_attempts: int = 6
    timeout_seconds: float = 1800.0


ToolFn = Callable[[dict[str, Any]], Awaitable[Any] | Any]


class AgentLoop:
    """Bounded model/tool loop used by workers.

    The loop is intentionally provider-agnostic. The model decides when to call a
    registered tool; the worker owns the actual filesystem/git/test operations.
    """

    def __init__(self, adapter: ModelAdapter, tools: dict[str, ToolFn], policy: AgentPolicy | None = None):
        self.adapter = adapter
        self.tools = tools
        self.policy = policy or AgentPolicy()

    async def run(self, messages: list[dict[str, str]], tool_specs: list[dict[str, Any]]) -> dict[str, Any]:
        started = time.monotonic()
        history = list(messages)
        repairs = 0
        steps = 0
        while steps < self.policy.max_steps:
            if time.monotonic() - started >= self.policy.timeout_seconds:
                return {"status": "checkpointed", "steps": steps, "repairs": repairs, "messages": history}
            steps += 1
            response: ModelResponse
            try:
                response = await self.adapter.generate(history, tools=tool_specs)
            except Exception as exc:
                if repairs < self.policy.repair_attempts:
                    repairs += 1
                    history.append({"role": "system", "content": f"Model call failed. Retry safely. Error: {exc}"})
                    continue
                return {"status": "failed", "steps": steps, "repairs": repairs, "error": str(exc), "messages": history}

            if response.text:
                history.append({"role": "assistant", "content": response.text})
            if not response.tool_calls:
                return {"status": "completed", "steps": steps, "repairs": repairs, "text": response.text, "usage": response.usage, "messages": history}

            for call in response.tool_calls:
                name = str(call.get("name", ""))
                args = call.get("arguments", {})
                if isinstance(args, str):
                    import json
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                if name not in self.tools:
                    result = {"error": f"unknown tool: {name}"}
                else:
                    try:
                        value = self.tools[name](args)
                        result = await value if hasattr(value, "__await__") else value
                    except Exception as exc:
                        repairs += 1
                        result = {"error": str(exc)}
                history.append({"role": "tool", "name": name, "content": str(result)})
                if repairs > self.policy.repair_attempts:
                    return {"status": "failed", "steps": steps, "repairs": repairs, "error": "self-repair limit exceeded", "messages": history}

        return {"status": "checkpointed", "steps": steps, "repairs": repairs, "messages": history}
