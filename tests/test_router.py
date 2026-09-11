import pytest

from agent_platform.router import ModelEndpoint, SmartRouter


def free(provider: str, model: str, **kwargs) -> ModelEndpoint:
    metadata = {"billing_type": "free", "zero_cost_verified": True, **kwargs.pop("metadata", {})}
    return ModelEndpoint(provider + "-" + model, provider, model, metadata=metadata, **kwargs)


def test_router_prefers_task_specific_endpoint():
    router = SmartRouter([
        free("p1", "general", task_fit=0.9, reliability=0.95, latency_ms=500),
        free("p2", "coder", task_fit=0.8, reliability=0.95, latency_ms=500, metadata={"task_fit": {"coding": 1.0}}),
    ])
    assert router.choose(tools=True, task_type="coding").model == "coder"


def test_router_prefers_role_specialist():
    router = SmartRouter([
        free("p1", "general", task_fit=1.0, reliability=0.95, metadata={"role_fit": {"coding": 0.6}}),
        free("p2", "coder", task_fit=0.8, reliability=0.95, metadata={"role_fit": {"coding": 1.0}}),
    ])
    assert router.choose(tools=True, task_type="coding", role="coding").model == "coder"


def test_router_avoids_rate_limited_endpoint_until_cooldown():
    router = SmartRouter([free("p1", "m1", reliability=1.0), free("p2", "m2", reliability=0.8)])
    router.mark_failure("p1-m1", "HTTP 429 Too Many Requests")
    assert router.choose().model == "m2"


def test_router_rejects_zero_quota():
    router = SmartRouter([free("p1", "empty", quota_remaining=0), free("p2", "ok", quota_remaining=1)])
    assert router.choose().model == "ok"


def test_provider_quota_exhaustion_quarantines_all_models_for_that_provider():
    router = SmartRouter([free("openrouter", "model-a"), free("openrouter", "model-b"), free("groq", "model-c")])
    router.mark_failure("openrouter-model-a", "Rate limit exceeded: free-models-per-day")
    assert router.endpoints[0].health == "QUOTA_EXHAUSTED"
    assert router.endpoints[1].health == "QUOTA_EXHAUSTED"
    assert router.choose().id == "groq-model-c"


def test_provider_quota_does_not_quarantine_other_free_providers():
    router = SmartRouter([free("openrouter", "model-a"), free("groq", "model-b")])
    router.mark_failure("openrouter-model-a", "daily quota exhausted")
    assert router.endpoints[0].health == "QUOTA_EXHAUSTED"
    assert router.endpoints[1].health == "ONLINE"
    assert router.choose().id == "groq-model-b"


def test_paid_and_unknown_endpoints_are_hard_excluded_even_if_better():
    router = SmartRouter([
        ModelEndpoint("paid", "paid-provider", "premium", reliability=1.0, task_fit=1.0, metadata={"billing_type": "paid"}),
        ModelEndpoint("unknown", "unknown-provider", "mystery", reliability=1.0, task_fit=1.0, metadata={"billing_type": "unknown"}),
        free("free-provider", "free-model", reliability=0.5),
    ])
    assert router.choose().id == "free-provider-free-model"
    assert all(SmartRouter.is_zero_cost(endpoint) is False for endpoint in router.endpoints[:2])


def test_candidate_free_status_without_verified_zero_cost_is_excluded():
    router = SmartRouter([
        ModelEndpoint("candidate", "provider", "candidate", metadata={"billing_type": "free", "free_status": "candidate"}),
        free("verified", "verified"),
    ])
    assert router.choose().id == "verified-verified"


def test_local_endpoint_is_optional_by_default():
    router = SmartRouter([
        ModelEndpoint("local", "ollama", "weak-local", task_fit=0.8, reliability=0.8, metadata={"billing_type": "local", "category": "local"}),
        free("remote", "strong-free", task_fit=1.0, reliability=1.0),
    ])
    assert router.choose().id == "remote-strong-free"


def test_local_can_be_explicitly_preferred(monkeypatch):
    monkeypatch.setenv("AGENT_PREFER_LOCAL", "true")
    router = SmartRouter([
        ModelEndpoint("local", "ollama", "local-model", task_fit=1.0, reliability=1.0, metadata={"billing_type": "local", "category": "local"}),
        free("remote", "remote-model", task_fit=1.0, reliability=0.99),
    ])
    assert router.choose().id == "local"


def test_local_endpoint_is_zero_cost_eligible():
    endpoint = ModelEndpoint("local", "ollama", "qwen", metadata={"billing_type": "local", "category": "local"})
    assert SmartRouter.is_zero_cost(endpoint)


def test_only_paid_or_unknown_endpoints_fail_cleanly():
    router = SmartRouter([
        ModelEndpoint("paid", "provider", "premium", metadata={"billing_type": "paid"}),
        ModelEndpoint("unknown", "provider2", "mystery", metadata={"billing_type": "unknown"}),
    ])
    with pytest.raises(RuntimeError, match="zero-cost"):
        router.choose()


def test_provider_diverse_pool_keeps_only_best_endpoint_per_provider():
    router = SmartRouter([
        free("openrouter", "best", reliability=0.99),
        free("openrouter", "worse", reliability=0.60),
        free("groq", "coder", reliability=0.90),
        ModelEndpoint("local", "ollama", "qwen", reliability=0.85, metadata={"billing_type": "local", "category": "local"}),
    ])
    pool = router.ranked_provider_diverse(task_type="coding", role="coding", max_providers=3)
    assert [item.provider for item in pool] == ["openrouter", "groq", "ollama"]
    assert [item.id for item in pool] == ["openrouter-best", "groq-coder", "local"]


def test_provider_diverse_pool_excludes_quarantined_provider():
    router = SmartRouter([free("openrouter", "a"), free("openrouter", "b"), free("groq", "c")])
    router.mark_failure("openrouter-a", "free-models-per-day quota exhausted")
    pool = router.ranked_provider_diverse(max_providers=5)
    assert [item.provider for item in pool] == ["groq"]
