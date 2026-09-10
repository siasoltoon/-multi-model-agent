from agent_platform.provider_api_catalog import VERIFIED_PROVIDER_APIS
from agent_platform.provider_registry import PROVIDER_REGISTRY


def test_verified_api_catalog_matches_registry():
    registry_ids = {item.provider_id for item in PROVIDER_REGISTRY}
    assert len(VERIFIED_PROVIDER_APIS) >= 15
    assert {item.provider_id for item in VERIFIED_PROVIDER_APIS}.issubset(registry_ids)


def test_verified_contracts_have_required_openai_paths():
    for contract in VERIFIED_PROVIDER_APIS:
        assert contract.verified is True
        assert contract.base_url.startswith("https://")
        assert contract.models_path == "/models"
        assert contract.chat_path == "/chat/completions"
