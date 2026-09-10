from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Category = Literal["direct", "gateway", "local"]
BillingType = Literal["permanent_free", "trial", "paid", "local", "unknown"]
FreeStatus = Literal["verified", "candidate", "expired", "unknown"]


@dataclass(frozen=True)
class ProviderDefinition:
    provider_id: str
    display_name: str
    category: Category
    base_url: str | None = None
    models_url: str | None = None
    api_key_env: str | None = None
    auth_scheme: str = "bearer"
    adapter: str = "openai_compatible"
    openai_compatible: bool = True
    billing_type: BillingType = "unknown"
    free_status: FreeStatus = "unknown"
    free_quota: str | None = None
    discovery_supported: bool = False
    tool_support: bool | None = None
    default_context_window: int = 32768
    role_fit: tuple[str, ...] = ()
    docs_url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "display_name": self.display_name,
            "category": self.category,
            "adapter": self.adapter,
            "openai_compatible": self.openai_compatible,
            "billing_type": self.billing_type,
            "free_status": self.free_status,
            "free_quota": self.free_quota,
            "discovery_supported": self.discovery_supported,
            "tool_support": self.tool_support,
            "default_context_window": self.default_context_window,
            "role_fit": self.role_fit,
            "docs_url": self.docs_url,
            **self.metadata,
        }


def _oc(
    provider_id: str,
    display_name: str,
    base_url: str,
    models_url: str,
    api_key_env: str,
    *,
    category: Category = "direct",
    billing_type: BillingType = "unknown",
    free_status: FreeStatus = "unknown",
    free_quota: str | None = None,
    context: int = 32768,
    roles: tuple[str, ...] = (),
    tool_support: bool | None = None,
    docs_url: str | None = None,
) -> ProviderDefinition:
    return ProviderDefinition(
        provider_id=provider_id, display_name=display_name, category=category,
        base_url=base_url, models_url=models_url, api_key_env=api_key_env,
        billing_type=billing_type, free_status=free_status, free_quota=free_quota,
        discovery_supported=True, tool_support=tool_support,
        default_context_window=context, role_fit=roles, docs_url=docs_url,
    )


# This is intentionally a capability catalog, not a claim that every candidate
# has an active free tier. Free/trial status is conservative and must be verified
# before an adapter is enabled automatically.
PROVIDER_REGISTRY: tuple[ProviderDefinition, ...] = (
    _oc("openrouter", "OpenRouter", "https://openrouter.ai/api/v1", "https://openrouter.ai/api/v1/models", "OPENROUTER_API_KEY", category="gateway", roles=("analysis", "coding", "testing", "review")),
    _oc("groq", "Groq", "https://api.groq.com/openai/v1", "https://api.groq.com/openai/v1/models", "GROQ_API_KEY", free_status="candidate", roles=("analysis", "coding", "testing")),
    _oc("mistral", "Mistral AI", "https://api.mistral.ai/v1", "https://api.mistral.ai/v1/models", "MISTRAL_API_KEY", roles=("analysis", "coding", "review")),
    _oc("cerebras", "Cerebras", "https://api.cerebras.ai/v1", "https://api.cerebras.ai/v1/models", "CEREBRAS_API_KEY", free_status="candidate", roles=("analysis", "coding")),
    _oc("together", "Together AI", "https://api.together.xyz/v1", "https://api.together.xyz/v1/models", "TOGETHER_API_KEY", free_status="candidate", roles=("coding", "testing")),
    _oc("fireworks", "Fireworks AI", "https://api.fireworks.ai/inference/v1", "https://api.fireworks.ai/inference/v1/models", "FIREWORKS_API_KEY", free_status="candidate", roles=("coding", "testing")),
    _oc("huggingface", "Hugging Face Inference", "https://router.huggingface.co/v1", "https://router.huggingface.co/v1/models", "HF_TOKEN", category="gateway", free_status="candidate", roles=("analysis", "coding", "testing")),
    _oc("chutes", "Chutes AI", "https://llm.chutes.ai/v1", "https://llm.chutes.ai/v1/models", "CHUTES_API_KEY", free_status="candidate", roles=("coding", "testing")),
    _oc("nebius", "Nebius AI Studio", "https://api.tokenfactory.nebius.com/v1", "https://api.tokenfactory.nebius.com/v1/models", "NEBIUS_API_KEY", free_status="candidate", roles=("coding", "analysis")),
    _oc("ovhcloud", "OVHcloud AI Endpoints", "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1", "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1/models", "OVHCLOUD_API_KEY", free_status="candidate", roles=("coding", "testing")),
    _oc("sambanova", "SambaNova", "https://api.sambanova.ai/v1", "https://api.sambanova.ai/v1/models", "SAMBANOVA_API_KEY", free_status="candidate", roles=("analysis", "coding")),
    _oc("friendli", "FriendliAI", "https://api.friendli.ai/serverless/v1", "https://api.friendli.ai/serverless/v1/models", "FRIENDLI_TOKEN", free_status="candidate", roles=("coding", "testing")),
    _oc("hyperbolic", "Hyperbolic", "https://api.hyperbolic.xyz/v1", "https://api.hyperbolic.xyz/v1/models", "HYPERBOLIC_API_KEY", free_status="candidate", roles=("coding", "analysis")),
    _oc("novita", "Novita AI", "https://api.novita.ai/openai", "https://api.novita.ai/openai/models", "NOVITA_API_KEY", free_status="candidate", roles=("coding", "testing")),
    _oc("siliconflow", "SiliconFlow", "https://api.siliconflow.com/v1", "https://api.siliconflow.com/v1/models", "SILICONFLOW_API_KEY", free_status="candidate", roles=("coding", "testing")),
    _oc("zhipu", "Zhipu AI / GLM", "https://open.bigmodel.cn/api/paas/v4", "https://open.bigmodel.cn/api/paas/v4/models", "ZHIPU_API_KEY", roles=("analysis", "coding", "review")),
    _oc("minimax", "MiniMax", "https://api.minimax.io/v1", "https://api.minimax.io/v1/models", "MINIMAX_API_KEY", free_status="candidate", roles=("coding", "analysis")),
    _oc("moonshot", "Moonshot / Kimi", "https://api.moonshot.ai/v1", "https://api.moonshot.ai/v1/models", "MOONSHOT_API_KEY", roles=("analysis", "coding", "review")),
    _oc("cohere", "Cohere", "https://api.cohere.com/compatibility/v1", "https://api.cohere.com/compatibility/v1/models", "COHERE_API_KEY", free_status="candidate", roles=("analysis", "review")),
    _oc("ai21", "AI21", "https://api.ai21.com/studio/v1", "https://api.ai21.com/studio/v1/models", "AI21_API_KEY", roles=("analysis", "review")),
    _oc("deepinfra", "DeepInfra", "https://api.deepinfra.com/v1/openai", "https://api.deepinfra.com/v1/openai/models", "DEEPINFRA_API_KEY", roles=("coding", "testing")),
    _oc("replicate", "Replicate", "https://api.replicate.com/v1", "https://api.replicate.com/v1/models", "REPLICATE_API_TOKEN", openai_compatible=False, roles=("coding", "testing")),
    _oc("modal", "Modal", "https://api.modal.com/v1", "https://api.modal.com/v1/models", "MODAL_TOKEN_ID", free_status="candidate", roles=("coding", "testing")),
    _oc("requesty", "Requesty", "https://router.requesty.ai/v1", "https://router.requesty.ai/v1/models", "REQUESTY_API_KEY", category="gateway", free_status="candidate", roles=("analysis", "coding", "testing")),
    _oc("portkey", "Portkey", "https://api.portkey.ai/v1", "https://api.portkey.ai/v1/models", "PORTKEY_API_KEY", category="gateway", roles=("analysis", "coding", "review")),
    _oc("featherless", "Featherless AI", "https://api.featherless.ai/v1", "https://api.featherless.ai/v1/models", "FEATHERLESS_API_KEY", free_status="candidate", roles=("coding", "testing")),
    _oc("vercel", "Vercel AI Gateway", "https://ai-gateway.vercel.sh/v1", "https://ai-gateway.vercel.sh/v1/models", "VERCEL_AI_GATEWAY_API_KEY", category="gateway", free_status="candidate", roles=("analysis", "coding", "testing")),
    _oc("nvidia", "NVIDIA NIM", "https://integrate.api.nvidia.com/v1", "https://integrate.api.nvidia.com/v1/models", "NVIDIA_API_KEY", free_status="candidate", roles=("coding", "analysis")),
    _oc("alibaba", "Alibaba Cloud DashScope", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/models", "DASHSCOPE_API_KEY", roles=("analysis", "coding")),
    _oc("routeway", "Routeway", "https://api.routeway.ai/v1", "https://api.routeway.ai/v1/models", "ROUTEWAY_API_KEY", category="gateway", free_status="candidate", roles=("analysis", "coding")),
    _oc("llmtr", "LLMTR", "https://api.llmtr.com/v1", "https://api.llmtr.com/v1/models", "LLMTR_API_KEY", category="gateway", free_status="candidate", roles=("analysis", "coding")),
    ProviderDefinition("ollama", "Ollama", "local", base_url="http://127.0.0.1:11434", models_url="http://127.0.0.1:11434/api/tags", api_key_env=None, adapter="ollama", openai_compatible=False, billing_type="local", free_status="verified", discovery_supported=False, tool_support=True, role_fit=("analysis", "coding", "testing")),
    ProviderDefinition("cloudflare_workers_ai", "Cloudflare Workers AI", "gateway", api_key_env="CLOUDFLARE_API_TOKEN", adapter="cloudflare", openai_compatible=False, billing_type="unknown", free_status="candidate", discovery_supported=False, role_fit=("coding", "testing")),
    ProviderDefinition("gemini", "Google Gemini / AI Studio", "direct", api_key_env="GEMINI_API_KEY", adapter="gemini", openai_compatible=False, billing_type="unknown", free_status="candidate", discovery_supported=False, role_fit=("analysis", "coding", "review", "verification")),
    ProviderDefinition("ai_horde", "AI Horde", "gateway", api_key_env="AI_HORDE_API_KEY", adapter="ai_horde", openai_compatible=False, billing_type="unknown", free_status="candidate", discovery_supported=False, role_fit=("analysis", "testing")),
)


PROVIDERS_BY_ID = {item.provider_id: item for item in PROVIDER_REGISTRY}


def get_provider(provider_id: str) -> ProviderDefinition:
    try:
        return PROVIDERS_BY_ID[provider_id]
    except KeyError as exc:
        raise KeyError(f"unknown provider: {provider_id}") from exc
