import asyncio

from agent_platform.adapters import FailoverAdapter, ModelResponse


class FailingAdapter:
    async def generate(self, messages, *, tools=None):
        raise RuntimeError("HTTP 429 Too Many Requests")


class WorkingAdapter:
    async def generate(self, messages, *, tools=None):
        return ModelResponse("ok", {}, {}, [])


def test_failover_switches_provider_after_failure():
    failures = []
    adapter = FailoverAdapter(
        [("bad", FailingAdapter()), ("good", WorkingAdapter())],
        on_failure=lambda endpoint_id, error: failures.append(endpoint_id),
    )
    result = asyncio.run(adapter.generate([{"role": "user", "content": "hi"}]))
    assert result.text == "ok"
    assert adapter.active_endpoint_id == "good"
    assert failures == ["bad"]
