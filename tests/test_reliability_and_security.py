import pytest

from agent_platform.reliability import checkpoint_payload, idempotency_key, redact_secrets, RetryPolicy
from agent_platform.workspace_tools import WorkspaceTools


def test_idempotency_key_is_stable_and_phase_specific():
    assert idempotency_key("t1", "coding") == idempotency_key("t1", "coding")
    assert idempotency_key("t1", "coding") != idempotency_key("t1", "testing")


def test_redaction_removes_common_secret_values():
    value = redact_secrets({"api_key": "super-secret", "text": "Bearer abc123"})
    assert value["api_key"] == "<redacted>"
    assert "abc123" not in value["text"]


def test_retry_policy_is_bounded():
    policy = RetryPolicy(attempts=3, base_delay=2, max_delay=5)
    assert policy.delay(1) == 2
    assert policy.delay(3) == 5
    assert policy.delay(99) == 5


def test_checkpoint_payload_is_versioned_and_redacted():
    payload = checkpoint_payload("t1", "coding", [{"role": "user", "content": "api_key=secret"}], steps=3, repairs=1)
    assert payload["version"] == 1
    assert payload["task_id"] == "t1"
    assert "secret" not in str(payload)


def test_workspace_rejects_escape(tmp_path):
    tools = WorkspaceTools(str(tmp_path))
    with pytest.raises(ValueError, match="escapes workspace"):
        tools.read_file({"path": "../outside.txt"})


def test_workspace_rejects_unsafe_package_flags(tmp_path):
    tools = WorkspaceTools(str(tmp_path))
    with pytest.raises(ValueError, match="blocked"):
        tools._argv(["pip", "install", "--system", "foo"])
