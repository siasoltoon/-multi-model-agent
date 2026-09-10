from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Multi-Model Agent"
    database_url: str = "sqlite:///./agent.db"
    redis_url: str = ""
    max_agent_steps: int = 64
    normal_agent_steps: int = 32
    self_repair_attempts: int = 6
    task_timeout_seconds: int = 1800
    worker_heartbeat_seconds: int = 15
    worker_lease_seconds: int = 300
    worker_poll_seconds: int = 5
    model_request_timeout_seconds: int = 180
    worker_auth_token: str = ""
    github_token: str = ""
    github_worker_repository: str = ""
    github_worker_workflow: str = "agent-worker.yml"
    github_worker_ref: str = "main"
    github_callback_token: str = ""
    public_base_url: str = ""
    callback_signature_tolerance_seconds: int = 300
    max_worker_attempts: int = 5
    model_failover_attempts: int = 3
    model_retry_attempts: int = 3
    model_retry_base_delay_seconds: float = 1.0
    model_retry_max_delay_seconds: float = 30.0
    model_max_response_bytes: int = 2_000_000
    model_max_tool_output_chars: int = 20_000
    model_config = SettingsConfigDict(env_prefix="AGENT_", env_file=".env", extra="ignore")


settings = Settings()
