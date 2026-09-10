from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderAPIContract:
    provider_id: str
    base_url: str
    models_path: str
    chat_path: str
    auth_scheme: str = "bearer"
    verified: bool = True


# Only contracts verified against current provider documentation are activated.
# The larger provider registry remains the catalog; this table controls live API use.
VERIFIED_PROVIDER_APIS: tuple[ProviderAPIContract, ...] = (
    ProviderAPIContract(
        "openrouter",
        "https://openrouter.ai/api/v1",
        "/models",
        "/chat/completions",
    ),
    ProviderAPIContract(
        "groq",
        "https://api.groq.com/openai/v1",
        "/models",
        "/chat/completions",
    ),
    ProviderAPIContract(
        "mistral",
        "https://api.mistral.ai/v1",
        "/models",
        "/chat/completions",
    ),
)

CONTRACTS_BY_PROVIDER = {item.provider_id: item for item in VERIFIED_PROVIDER_APIS}


def get_api_contract(provider_id: str) -> ProviderAPIContract | None:
    return CONTRACTS_BY_PROVIDER.get(provider_id)


def is_api_verified(provider_id: str) -> bool:
    contract = get_api_contract(provider_id)
    return bool(contract and contract.verified)
