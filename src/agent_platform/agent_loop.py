from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .adapters import ModelAdapter, ModelResponse
from .context_saver import ContextSaver


@dataclass
class AgentPolicy:
    max_steps: int = 64
    repair_attempts: int = 6
    timeout_seconds: float = 1800.0


ToolFn = Callable[[dict[str, Any]], Awaitable[Any] | Any]


class AgentLoop:
    """Provider-neutral coding loop with correct tool-call protocol and bounded repair."""

    def __init__(self, adapter: ModelAdapter, tools: dict[str, ToolFn], policy: AgentPolicy | None = None):
        self.adapter = adapter
        self.tools = tools
        self.policy = policy or AgentPolicy()
        self.context_saver = ContextSaver()

    async def run(self, messages: list[dict[str, Any]], tool_specs: list[dict[str, Any]]) -> dict[str, Any]:
        started = time.monotonic()
        history = list(messages)
        repairs = 0
        steps = 0
        while steps < self.policy.max_steps:
            if time.monotonic() - started >= self.policy.timeout_seconds:
                return {"status": "checkpointed", "steps": steps, "repairs": repairs, "messages": history}
            model_history = self.context_saver.compact(history)
            try:
                response: ModelResponse = await self.adapter.generate(model_history, tools=tool_specs)
            except Exception as exc:
                # Provider/model retries are recovery attempts, not productive
                # agent steps. Do not let a transient gateway failure consume a
                # tiny role budget (for example analysis may only have 2 steps).
                repairs += 1
                history.append({"role": "system", "content": f"Model call failed. Retry safely. Error: {exc}"})
                if repairs >= self.policy.repair_attempts:
                    return {"status": "failed", "steps": steps, "repairs": repairs, "error": str(exc), "messages": history}
                continue

            steps += 1
            assistant = {"role": "assistant", "content": response.text or None}
            if response.tool_calls:
                assistant["tool_calls"] = [
                    {"id": c.get("id") or f"call_{i}", "type": "function", "function": {
                        "name": str(c.get("name", "")), "arguments": c.get("arguments", "{}") if isinstance(c.get("arguments", "{}"), str) else json.dumps(c.get("arguments", {}))
                    }} for i, c in enumerate(response.tool_calls)
                ]
            history.append(assistant)
            if not response.tool_calls:
                return {"status": "completed", "steps": steps, "repairs": repairs, "text": response.text, "usage": response.usage, "messages": history}

            for call in response.tool_calls:
                call_id = str(call.get("id") or "")
                name = str(call.get("name", ""))
                args = call.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                try:
                    if name not in self.tools:
                        raise ValueError(f"unknown tool: {name}")
                    value = self.tools[name](args)
                    result = await value if hasattr(value, "__await__") else value
                except Exception as exc:
                    repairs += 1
                    result = {"error": str(exc)}
                history.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(result, ensure_ascii=False, default=str)})
                if repairs >= self.policy.repair_attempts:
                    return {"status": "failed", "steps": steps, "repairs": repairs, "error": "self-repair limit exceeded", "messages": history}

        return {"status": "checkpointed", "steps": steps, "repairs": repairs, "messages": history}
