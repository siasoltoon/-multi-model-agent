import pytest

from agent_platform.provider_api_catalog import VERIFIED_PROVIDER_APIS
from agent_platform.provider_connections import build_provider_adapter
from agent_platform.provider_registry import get_provider


@pytest.mark.parametrize("contract", VERIFIED_PROVIDER_APIS, ids=lambda item: item.provider_id)
def test_every_verified_provider_has_a_connectable_shared_adapter(monkeypatch, contract):
    definition = get_provider(contract.provider_id)
    assert definition.openai_compatible is True
    assert contract.verified is True
    env_name = definition.api_key_env
    assert env_name
    monkeypatch.setenv(env_name, "test-key")
    adapter = build_provider_adapter(contract.provider_id, model="test-model")
    assert adapter.url == contract.base_url.rstrip("/") + "/chat/completions"
    assert adapter.api_key == "test-key"
    assert adapter.model == "test-model"
