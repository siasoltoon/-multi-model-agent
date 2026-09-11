import asyncio

from agent_platform.adapters import FailoverAdapter, ModelResponse, ProviderFailoverExhausted


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


def test_failover_stops_on_provider_quota_exhaustion():
    calls = []

    class QuotaFailingAdapter:
        async def generate(self, messages, *, tools=None):
            calls.append("quota")
            raise RuntimeError("Rate limit exceeded: free-models-per-day")

    class ShouldNotRunAdapter:
        async def generate(self, messages, *, tools=None):
            calls.append("sibling")
            return ModelResponse("unexpected", {}, {}, [])

    adapter = FailoverAdapter(
        [("openrouter-a", QuotaFailingAdapter()), ("openrouter-b", ShouldNotRunAdapter())],
        on_failure=lambda endpoint_id, error: "free-models-per-day" in str(error),
        quarantine_on_failure=True,
    )

    async def run():
        try:
            await adapter.generate([{"role": "user", "content": "quota"}])
        except ProviderFailoverExhausted as exc:
            return exc
        raise AssertionError("expected provider failover exhaustion")

    error = asyncio.run(run())
    assert error.endpoint_ids == ("openrouter-a",)
    assert calls == ["quota"]


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


def test_quarantine_persists_across_model_calls_and_uses_next_endpoint():
    class CountingFailingAdapter:
        def __init__(self):
            self.calls = 0

        async def generate(self, messages, *, tools=None):
            self.calls += 1
            raise RuntimeError("provider returned error")

    bad = CountingFailingAdapter()
    good = WorkingAdapter()
    adapter = FailoverAdapter(
        [("bad", bad), ("good", good)],
        quarantine_on_failure=True,
    )

    async def run():
        first = await adapter.generate([{"role": "user", "content": "first"}])
        second = await adapter.generate([{"role": "user", "content": "second"}])
        return first, second

    first, second = asyncio.run(run())
    assert first.text == "ok"
    assert second.text == "ok"
    assert bad.calls == 1
    assert adapter.active_endpoint_id == "good"


def test_quarantine_reports_exhaustion_without_retrying_failed_endpoints():
    bad = FailingAdapter()
    adapter = FailoverAdapter([("bad", bad)], quarantine_on_failure=True)

    async def run():
        try:
            await adapter.generate([{"role": "user", "content": "first"}])
        except ProviderFailoverExhausted as exc:
            return exc
        raise AssertionError("expected provider failover exhaustion")

    error = asyncio.run(run())
    assert error.endpoint_ids == ("bad",)
    assert "failover exhausted" in str(error)
