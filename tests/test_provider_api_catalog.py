from agent_platform.provider_api_catalog import VERIFIED_PROVIDER_APIS, get_api_contract, is_api_verified


def test_verified_api_catalog_contains_first_batch():
    ids = {item.provider_id for item in VERIFIED_PROVIDER_APIS}
    assert {"openrouter", "groq", "mistral"}.issubset(ids)


def test_verified_contracts_use_expected_openai_paths():
    for item in VERIFIED_PROVIDER_APIS:
        assert item.verified is True
        assert item.models_path == "/models"
        assert item.chat_path == "/chat/completions"
        assert item.base_url.startswith("https://")


def test_api_activation_is_explicit():
    assert is_api_verified("openrouter") is True
    assert is_api_verified("cerebras") is False
    assert get_api_contract("cerebras") is None
