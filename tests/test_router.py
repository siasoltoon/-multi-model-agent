from agent_platform.router import ModelEndpoint, SmartRouter


def test_router_prefers_task_specific_endpoint():
    router = SmartRouter([
        ModelEndpoint("general", "p1", "general", task_fit=0.9, reliability=0.95, latency_ms=500),
        ModelEndpoint("coding", "p2", "coder", task_fit=0.8, reliability=0.95, latency_ms=500,
                      metadata={"task_fit": {"coding": 1.0}}),
    ])
    assert router.choose(tools=True, task_type="coding").id == "coding"


def test_router_prefers_role_specialist():
    router = SmartRouter([
        ModelEndpoint("general", "p1", "general", task_fit=1.0, reliability=0.95,
                      metadata={"role_fit": {"coding": 0.6}}),
        ModelEndpoint("coder", "p2", "coder", task_fit=0.8, reliability=0.95,
                      metadata={"role_fit": {"coding": 1.0}}),
    ])
    assert router.choose(tools=True, task_type="coding", role="coding").id == "coder"


def test_router_avoids_rate_limited_endpoint_until_cooldown():
    router = SmartRouter([
        ModelEndpoint("limited", "p1", "m1", reliability=1.0),
        ModelEndpoint("healthy", "p2", "m2", reliability=0.8),
    ])
    router.mark_failure("limited", "HTTP 429 Too Many Requests")
    assert router.choose().id == "healthy"


def test_router_rejects_zero_quota():
    router = SmartRouter([
        ModelEndpoint("empty", "p1", "m1", quota_remaining=0),
        ModelEndpoint("ok", "p2", "m2", quota_remaining=1),
    ])
    assert router.choose().id == "ok"


def test_provider_quota_exhaustion_quarantines_all_models_for_that_provider():
    router = SmartRouter([
        ModelEndpoint("or-a", "openrouter", "model-a", metadata={"billing_type": "free"}),
        ModelEndpoint("or-b", "openrouter", "model-b", metadata={"billing_type": "free"}),
        ModelEndpoint("groq", "groq", "model-c", metadata={"billing_type": "paid_or_unknown"}),
    ])
    router.mark_failure("or-a", "Rate limit exceeded: free-models-per-day")
    assert router.endpoints[0].health == "QUOTA_EXHAUSTED"
    assert router.endpoints[1].health == "QUOTA_EXHAUSTED"
    assert router.choose().id == "groq"


def test_provider_quota_does_not_quarantine_other_providers():
    router = SmartRouter([
        ModelEndpoint("or-a", "openrouter", "model-a"),
        ModelEndpoint("groq", "groq", "model-b"),
    ])
    router.mark_failure("or-a", "daily quota exhausted")
    assert router.endpoints[0].health == "QUOTA_EXHAUSTED"
    assert router.endpoints[1].health == "ONLINE"
    assert router.choose().id == "groq"


def test_free_provider_is_not_preferred_over_paid_or_unknown():
    router = SmartRouter([
        ModelEndpoint("free", "openrouter", "free", reliability=0.95, metadata={"billing_type": "free"}),
        ModelEndpoint("paid", "groq", "paid", reliability=0.95, metadata={"billing_type": "paid_or_unknown"}),
    ])
    assert router.choose().id == "paid"
