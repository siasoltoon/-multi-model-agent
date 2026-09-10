from __future__ import annotations

import os

from .adapters import OpenAICompatibleAdapter
from .native_adapters import build_native_adapter
from .provider_api_catalog import get_api_contract
from .provider_registry import get_provider


def _credential(provider_id: str, api_key: str | None) -> str:
    definition = get_provider(provider_id)
    credential = api_key if api_key is not None else os.getenv(definition.api_key_env or "", "")
    credential = credential.strip()
    if not credential:
        raise RuntimeError(f"missing provider credential: {definition.api_key_env or provider_id}")
    return credential


def build_provider_adapter(provider_id: str, *, model: str, timeout: float = 180.0, api_key: str | None = None):
    """Build the correct adapter for a provider; remote activation requires a verified contract."""
    definition = get_provider(provider_id)
    credential = _credential(provider_id, api_key)
    if definition.adapter in {"openai", "anthropic", "gemini"}:
        return build_native_adapter(provider_id, api_key=credential, model=model, timeout=timeout)
    contract = get_api_contract(provider_id)
    if contract is None or not contract.verified:
        raise ValueError(f"provider API is not verified: {provider_id}")
    if not definition.openai_compatible:
        raise ValueError(f"provider requires a native adapter: {provider_id}")
    return OpenAICompatibleAdapter(contract.base_url, credential, model, timeout=timeout)


def verified_provider_ids() -> tuple[str, ...]:
    from .provider_api_catalog import VERIFIED_PROVIDER_APIS
    return tuple(item.provider_id for item in VERIFIED_PROVIDER_APIS if item.verified)
