from agent_platform.discovery import BUILTIN_PROVIDERS
from agent_platform.provider_api_catalog import get_api_contract
from agent_platform.provider_registry import get_provider


def test_tokenharbor_is_verified_openai_compatible_provider():
    provider = get_provider("tokenharbor")
    contract = get_api_contract("tokenharbor")
    specs = {item.name: item for item in BUILTIN_PROVIDERS}

    assert provider.api_key_env == "TOKENHARBOR_API_KEY"
    assert provider.openai_compatible is True
    assert provider.free_status == "verified"
    assert provider.free_quota == "rolling_7_day_value_allowance"
    assert contract is not None
    assert contract.base_url == "https://tokenharbor.ai/v1"
    assert contract.models_path == "/models"
    assert contract.chat_path == "/chat/completions"
    assert specs["tokenharbor"].models_url == "https://tokenharbor.ai/v1/models"
