from agent_platform.router import ModelEndpoint, SmartRouter


def test_router_prefers_task_specific_endpoint():
    router = SmartRouter([
        ModelEndpoint("general", "p1", "general", task_fit=0.9, reliability=0.95, latency_ms=500),
        ModelEndpoint("coding", "p2", "coder", task_fit=0.8, reliability=0.95, latency_ms=500,
                      metadata={"task_fit": {"coding": 1.0}}),
    ])
    assert router.choose(tools=True, task_type="coding").id == "coding"


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
