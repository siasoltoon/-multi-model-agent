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


def test_failover_allows_later_recovery_retry_of_same_endpoint():
    class RecoveringAdapter:
        def __init__(self):
            self.calls = 0

        async def generate(self, messages, *, tools=None):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary gateway failure")
            return ModelResponse("recovered", {}, {}, [])

    failures = []
    provider = RecoveringAdapter()
    adapter = FailoverAdapter(
        [("only", provider)],
        on_failure=lambda endpoint_id, error: failures.append(endpoint_id),
    )

    async def run():
        try:
            await adapter.generate([{"role": "user", "content": "first"}])
        except RuntimeError:
            pass
        return await adapter.generate([{"role": "user", "content": "retry"}])

    result = asyncio.run(run())

    assert result.text == "recovered"
    assert provider.calls == 2
    assert failures == ["only"]
