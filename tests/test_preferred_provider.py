from types import SimpleNamespace

from agent_platform.subtask_executor import SubtaskExecutor


def test_preferred_provider_is_promoted(monkeypatch):
    endpoints = [
        SimpleNamespace(provider="openrouter", model="openrouter/free", id="or"),
        SimpleNamespace(provider="tokenharbor", model="deepseek-v4.1-flash:free", id="th"),
    ]

    class FakeRouter:
        def ranked_provider_diverse(self, **kwargs):
            return list(endpoints)

    monkeypatch.setenv("AGENT_PREFERRED_PROVIDER", "tokenharbor")
    executor = SubtaskExecutor(FakeRouter(), "/tmp")
    monkeypatch.setattr("agent_platform.subtask_executor._build_adapter", lambda endpoint, timeout: object())

    adapter, selected = executor._adapter("coder")

    assert selected[0].provider == "tokenharbor"
    assert selected[0].model == "deepseek-v4.1-flash:free"
    assert len(selected) == 2
    assert adapter is not None
