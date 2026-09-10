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


# Only providers with a verified OpenAI-compatible contract are activated here.
# The larger provider registry remains the catalog; this table controls live API use.
VERIFIED_PROVIDER_APIS: tuple[ProviderAPIContract, ...] = (
    ProviderAPIContract("openrouter", "https://openrouter.ai/api/v1", "/models", "/chat/completions"),
    ProviderAPIContract("groq", "https://api.groq.com/openai/v1", "/models", "/chat/completions"),
    ProviderAPIContract("mistral", "https://api.mistral.ai/v1", "/models", "/chat/completions"),
    ProviderAPIContract("deepseek", "https://api.deepseek.com", "/models", "/chat/completions"),
    ProviderAPIContract("together", "https://api.together.xyz/v1", "/models", "/chat/completions"),
    ProviderAPIContract("fireworks", "https://api.fireworks.ai/inference/v1", "/models", "/chat/completions"),
    ProviderAPIContract("cerebras", "https://api.cerebras.ai/v1", "/models", "/chat/completions"),
    ProviderAPIContract("friendli", "https://api.friendli.ai/serverless/v1", "/models", "/chat/completions"),
    ProviderAPIContract("hyperbolic", "https://api.hyperbolic.xyz/v1", "/models", "/chat/completions"),
    ProviderAPIContract("novita", "https://api.novita.ai/openai/v1", "/models", "/chat/completions"),
    ProviderAPIContract("moonshot", "https://api.moonshot.ai/v1", "/models", "/chat/completions"),
    ProviderAPIContract("minimax", "https://api.minimax.io/v1", "/models", "/chat/completions"),
    ProviderAPIContract("zhipu", "https://open.bigmodel.cn/api/paas/v4", "/models", "/chat/completions"),
    ProviderAPIContract("alibaba", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "/models", "/chat/completions"),
    ProviderAPIContract("siliconflow", "https://api.siliconflow.com/v1", "/models", "/chat/completions"),
    ProviderAPIContract("deepinfra", "https://api.deepinfra.com/v1/openai", "/models", "/chat/completions"),
    ProviderAPIContract("chutes", "https://llm.chutes.ai/v1", "/models", "/chat/completions"),
)

CONTRACTS_BY_PROVIDER = {item.provider_id: item for item in VERIFIED_PROVIDER_APIS}


def get_api_contract(provider_id: str) -> ProviderAPIContract | None:
    return CONTRACTS_BY_PROVIDER.get(provider_id)


def is_api_verified(provider_id: str) -> bool:
    contract = get_api_contract(provider_id)
    return bool(contract and contract.verified)
