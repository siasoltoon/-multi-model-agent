from __future__ import annotations

import os

from .adapters import OpenAICompatibleAdapter
from .provider_api_catalog import get_api_contract
from .provider_registry import get_provider


def build_provider_adapter(provider_id: str, *, model: str, timeout: float = 180.0, api_key: str | None = None) -> OpenAICompatibleAdapter:
    """Build a runtime adapter only for an explicitly verified provider contract."""
    definition = get_provider(provider_id)
    contract = get_api_contract(provider_id)
    if contract is None or not contract.verified:
        raise ValueError(f"provider API is not verified: {provider_id}")
    if not definition.openai_compatible:
        raise ValueError(f"provider requires a native adapter: {provider_id}")
    credential = api_key if api_key is not None else os.getenv(definition.api_key_env or "", "")
    credential = credential.strip()
    if not credential:
        raise RuntimeError(f"missing provider credential: {definition.api_key_env or provider_id}")
    return OpenAICompatibleAdapter(
        base_url=contract.base_url,
        api_key=credential,
        model=model,
        timeout=timeout,
    )


def verified_provider_ids() -> tuple[str, ...]:
    from .provider_api_catalog import VERIFIED_PROVIDER_APIS

    return tuple(item.provider_id for item in VERIFIED_PROVIDER_APIS if item.verified)
