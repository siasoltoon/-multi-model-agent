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


def _oc(provider_id: str, display_name: str, base_url: str, models_url: str | None, api_key_env: str, *, category: Category = "direct", billing_type: BillingType = "unknown", free_status: FreeStatus = "unknown", free_quota: str | None = None, context: int = 32768, roles: tuple[str, ...] = (), tool_support: bool | None = None, docs_url: str | None = None, discovery_supported: bool = False) -> ProviderDefinition:
    return ProviderDefinition(provider_id=provider_id, display_name=display_name, category=category, base_url=base_url, models_url=models_url, api_key_env=api_key_env, billing_type=billing_type, free_status=free_status, free_quota=free_quota, discovery_supported=discovery_supported, tool_support=tool_support, default_context_window=context, role_fit=roles, docs_url=docs_url)


# Catalog first, API activation second. Only explicitly verified providers have
# discovery_supported=True; catalog entries alone must never cause network calls.
PROVIDER_REGISTRY: tuple[ProviderDefinition, ...] = (
    # Gateways / aggregators
    _oc("openrouter", "OpenRouter", "https://openrouter.ai/api/v1", "https://openrouter.ai/api/v1/models", "OPENROUTER_API_KEY", category="gateway", roles=("analysis", "architecture", "coding", "testing", "review", "security", "verification"), discovery_supported=True),
    _oc("tokenharbor", "Token Harbor", "https://tokenharbor.ai/v1", "https://tokenharbor.ai/v1/models", "TOKENHARBOR_API_KEY", category="gateway", free_status="verified", free_quota="rolling_7_day_value_allowance", roles=("analysis", "architecture", "coding", "testing", "review", "security", "verification"), discovery_supported=True),
    _oc("requesty", "Requesty", "https://router.requesty.ai/v1", "https://router.requesty.ai/v1/models", "REQUESTY_API_KEY", category="gateway", free_status="candidate", roles=("analysis", "coding", "testing")),
    _oc("portkey", "Portkey", "https://api.portkey.ai/v1", "https://api.portkey.ai/v1/models", "PORTKEY_API_KEY", category="gateway", roles=("analysis", "architecture", "coding", "review")),
    _oc("vercel", "Vercel AI Gateway", "https://ai-gateway.vercel.sh/v1", "https://ai-gateway.vercel.sh/v1/models", "VERCEL_AI_GATEWAY_API_KEY", category="gateway", free_status="candidate", roles=("analysis", "coding", "testing")),
    _oc("routeway", "Routeway", "https://api.routeway.ai/v1", "https://api.routeway.ai/v1/models", "ROUTEWAY_API_KEY", category="gateway", free_status="candidate", roles=("analysis", "coding")),
    _oc("llmtr", "LLMTR", "https://api.llmtr.com/v1", "https://api.llmtr.com/v1/models", "LLMTR_API_KEY", category="gateway", free_status="candidate", roles=("analysis", "coding")),
    _oc("huggingface", "Hugging Face Inference", "https://router.huggingface.co/v1", "https://router.huggingface.co/v1/models", "HF_TOKEN", category="gateway", free_status="candidate", roles=("analysis", "coding", "testing")),
    _oc("featherless", "Featherless AI", "https://api.featherless.ai/v1", "https://api.featherless.ai/v1/models", "FEATHERLESS_API_KEY", free_status="candidate", roles=("coding", "testing")),
    _oc("deepinfra", "DeepInfra", "https://api.deepinfra.com/v1/openai", "https://api.deepinfra.com/v1/openai/models", "DEEPINFRA_API_KEY", roles=("coding", "testing")),
    _oc("siliconflow", "SiliconFlow", "https://api.siliconflow.com/v1", "https://api.siliconflow.com/v1/models", "SILICONFLOW_API_KEY", free_status="candidate", roles=("coding", "testing")),
    # Direct OpenAI-compatible providers
    _oc("groq", "Groq", "https://api.groq.com/openai/v1", "https://api.groq.com/openai/v1/models", "GROQ_API_KEY", free_status="candidate", roles=("analysis", "coding", "testing"), discovery_supported=True),
    _oc("mistral", "Mistral AI", "https://api.mistral.ai/v1", "https://api.mistral.ai/v1/models", "MISTRAL_API_KEY", roles=("analysis", "coding", "review"), discovery_supported=True),
    _oc("cerebras", "Cerebras", "https://api.cerebras.ai/v1", "https://api.cerebras.ai/v1/models", "CEREBRAS_API_KEY", free_status="candidate", roles=("analysis", "coding"), discovery_supported=True),
    _oc("together", "Together AI", "https://api.together.xyz/v1", "https://api.together.xyz/v1/models", "TOGETHER_API_KEY", free_status="candidate", roles=("coding", "testing"), discovery_supported=True),
    _oc("fireworks", "Fireworks AI", "https://api.fireworks.ai/inference/v1", "https://api.fireworks.ai/inference/v1/models", "FIREWORKS_API_KEY", free_status="candidate", roles=("coding", "testing"), discovery_supported=True),
    _oc("chutes", "Chutes AI", "https://llm.chutes.ai/v1", "https://llm.chutes.ai/v1/models", "CHUTES_API_KEY", free_status="candidate", roles=("coding", "testing")),
    _oc("nebius", "Nebius AI Studio", "https://api.tokenfactory.nebius.com/v1", "https://api.tokenfactory.nebius.com/v1/models", "NEBIUS_API_KEY", free_status="candidate", roles=("coding", "analysis")),
    _oc("ovhcloud", "OVHcloud AI Endpoints", "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1", "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1/models", "OVHCLOUD_API_KEY", free_status="candidate", roles=("coding", "testing")),
    _oc("sambanova", "SambaNova", "https://api.sambanova.ai/v1", "https://api.sambanova.ai/v1/models", "SAMBANOVA_API_KEY", free_status="candidate", roles=("analysis", "coding")),
    _oc("friendli", "FriendliAI", "https://api.friendli.ai/serverless/v1", "https://api.friendli.ai/serverless/v1/models", "FRIENDLI_TOKEN", free_status="candidate", roles=("coding", "testing")),
    _oc("hyperbolic", "Hyperbolic", "https://api.hyperbolic.xyz/v1", "https://api.hyperbolic.xyz/v1/models", "HYPERBOLIC_API_KEY", free_status="candidate", roles=("coding", "analysis")),
    _oc("novita", "Novita AI", "https://api.novita.ai/openai", "https://api.novita.ai/openai/models", "NOVITA_API_KEY", free_status="candidate", roles=("coding", "testing")),
    _oc("minimax", "MiniMax", "https://api.minimax.io/v1", "https://api.minimax.io/v1/models", "MINIMAX_API_KEY", free_status="candidate", roles=("coding", "analysis")),
    _oc("moonshot", "Moonshot / Kimi", "https://api.moonshot.ai/v1", "https://api.moonshot.ai/v1/models", "MOONSHOT_API_KEY", roles=("analysis", "coding", "review")),
    _oc("cohere", "Cohere", "https://api.cohere.com/compatibility/v1", "https://api.cohere.com/compatibility/v1/models", "COHERE_API_KEY", free_status="candidate", roles=("analysis", "review")),
    _oc("ai21", "AI21", "https://api.ai21.com/studio/v1", "https://api.ai21.com/studio/v1/models", "AI21_API_KEY", roles=("analysis", "review")),
    _oc("zhipu", "Zhipu AI / GLM", "https://open.bigmodel.cn/api/paas/v4", "https://open.bigmodel.cn/api/paas/v4/models", "ZHIPU_API_KEY", roles=("analysis", "coding", "review")),
    _oc("alibaba", "Alibaba Cloud DashScope", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/models", "DASHSCOPE_API_KEY", roles=("analysis", "coding")),
    _oc("nvidia", "NVIDIA NIM", "https://integrate.api.nvidia.com/v1", "https://integrate.api.nvidia.com/v1/models", "NVIDIA_API_KEY", free_status="candidate", roles=("coding", "analysis")),
    # Native providers
    ProviderDefinition("openai", "OpenAI", "direct", api_key_env="OPENAI_API_KEY", adapter="openai", role_fit=("analysis", "architecture", "coding", "testing", "review", "security", "verification")),
    ProviderDefinition("anthropic", "Anthropic", "direct", api_key_env="ANTHROPIC_API_KEY", adapter="anthropic", openai_compatible=False, role_fit=("analysis", "architecture", "coding", "review", "security")),
    ProviderDefinition("gemini", "Google Gemini / AI Studio", "direct", api_key_env="GEMINI_API_KEY", adapter="gemini", openai_compatible=False, free_status="candidate", role_fit=("analysis", "architecture", "coding", "review", "verification")),
    ProviderDefinition("deepseek", "DeepSeek", "direct", base_url="https://api.deepseek.com", models_url="https://api.deepseek.com/models", api_key_env="DEEPSEEK_API_KEY", adapter="deepseek", role_fit=("analysis", "coding", "review")),
    ProviderDefinition("xai", "xAI", "direct", base_url="https://api.x.ai/v1", models_url="https://api.x.ai/v1/models", api_key_env="XAI_API_KEY", adapter="xai", role_fit=("analysis", "coding", "review")),
    ProviderDefinition("perplexity", "Perplexity", "direct", base_url="https://api.perplexity.ai", models_url="https://api.perplexity.ai/models", api_key_env="PERPLEXITY_API_KEY", adapter="perplexity", role_fit=("analysis", "research", "verification")),
    ProviderDefinition("cloudflare_workers_ai", "Cloudflare Workers AI", "gateway", api_key_env="CLOUDFLARE_API_TOKEN", adapter="cloudflare", openai_compatible=False, free_status="candidate", role_fit=("coding", "testing")),
    ProviderDefinition("aws_bedrock", "Amazon Bedrock", "direct", api_key_env="AWS_ACCESS_KEY_ID", adapter="aws_bedrock", openai_compatible=False, role_fit=("analysis", "coding", "testing")),
    ProviderDefinition("google_vertex", "Google Vertex AI", "direct", api_key_env="GOOGLE_APPLICATION_CREDENTIALS", adapter="vertex", openai_compatible=False, role_fit=("analysis", "coding", "review")),
    ProviderDefinition("azure_foundry", "Microsoft Foundry / Azure AI", "direct", api_key_env="AZURE_OPENAI_API_KEY", adapter="azure", openai_compatible=False, role_fit=("analysis", "coding", "review")),
    ProviderDefinition("ibm_watsonx", "IBM watsonx", "direct", api_key_env="WATSONX_API_KEY", adapter="watsonx", openai_compatible=False, role_fit=("analysis", "review")),
    ProviderDefinition("oracle_genai", "Oracle Generative AI", "direct", api_key_env="OCI_CONFIG_FILE", adapter="oracle_genai", openai_compatible=False, role_fit=("analysis", "coding")),
    ProviderDefinition("sap_ai_core", "SAP AI Core", "direct", api_key_env="SAP_AI_CORE_CLIENT_ID", adapter="sap_ai_core", openai_compatible=False, role_fit=("analysis", "coding")),
    ProviderDefinition("databricks", "Databricks Model Serving", "direct", api_key_env="DATABRICKS_TOKEN", adapter="databricks", role_fit=("analysis", "coding")),
    ProviderDefinition("snowflake", "Snowflake Cortex", "direct", api_key_env="SNOWFLAKE_ACCOUNT", adapter="snowflake", openai_compatible=False, role_fit=("analysis", "verification")),
    # Hosted inference / compute
    ProviderDefinition("replicate", "Replicate", "direct", base_url="https://api.replicate.com/v1", api_key_env="REPLICATE_API_TOKEN", adapter="replicate", openai_compatible=False, role_fit=("coding", "testing")),
    ProviderDefinition("modal", "Modal", "direct", api_key_env="MODAL_TOKEN_ID", adapter="modal", openai_compatible=False, free_status="candidate", role_fit=("coding", "testing")),
    ProviderDefinition("ai_horde", "AI Horde", "gateway", api_key_env="AI_HORDE_API_KEY", adapter="ai_horde", openai_compatible=False, free_status="candidate", role_fit=("analysis", "testing")),
    ProviderDefinition("pollinations", "Pollinations AI", "gateway", api_key_env="POLLINATIONS_API_KEY", adapter="pollinations", openai_compatible=False, free_status="candidate", role_fit=("analysis", "testing")),
    # Local runtimes
    ProviderDefinition("ollama", "Ollama", "local", base_url="http://127.0.0.1:11434", models_url="http://127.0.0.1:11434/api/tags", adapter="ollama", openai_compatible=False, billing_type="local", free_status="verified", tool_support=True, role_fit=("analysis", "architecture", "coding", "testing", "review")),
    ProviderDefinition("vllm", "vLLM", "local", adapter="vllm", billing_type="local", free_status="verified", role_fit=("analysis", "coding", "testing")),
    ProviderDefinition("localai", "LocalAI", "local", adapter="localai", billing_type="local", free_status="verified", role_fit=("analysis", "coding", "testing")),
    ProviderDefinition("lmstudio", "LM Studio", "local", adapter="lmstudio", billing_type="local", free_status="verified", role_fit=("analysis", "coding", "testing")),
    ProviderDefinition("llamacpp", "llama.cpp server", "local", adapter="llamacpp", billing_type="local", free_status="verified", role_fit=("analysis", "coding", "testing")),
    ProviderDefinition("sglang", "SGLang", "local", adapter="sglang", billing_type="local", free_status="verified", role_fit=("analysis", "coding", "testing")),
    ProviderDefinition("tgi", "Text Generation Inference", "local", adapter="tgi", billing_type="local", free_status="verified", role_fit=("coding", "testing")),
    ProviderDefinition("jan", "Jan", "local", adapter="jan", billing_type="local", free_status="verified", role_fit=("analysis", "coding")),
    ProviderDefinition("gpt4all", "GPT4All", "local", adapter="gpt4all", billing_type="local", free_status="verified", role_fit=("analysis", "coding")),
    ProviderDefinition("koboldcpp", "KoboldCpp", "local", adapter="koboldcpp", billing_type="local", free_status="verified", role_fit=("analysis", "coding")),
)

PROVIDERS_BY_ID: dict[str, ProviderDefinition] = {item.provider_id: item for item in PROVIDER_REGISTRY}


def get_provider(provider_id: str) -> ProviderDefinition:
    try:
        return PROVIDERS_BY_ID[provider_id]
    except KeyError as exc:
        raise KeyError(f"Unknown provider: {provider_id}") from exc
