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
    model_request_timeout_seconds: int = 180
    worker_auth_token: str = ""
    github_token: str = ""
    github_worker_repository: str = ""
    github_worker_workflow: str = "agent-worker.yml"
    github_worker_ref: str = "main"
    github_callback_token: str = ""
    public_base_url: str = ""
    model_config = SettingsConfigDict(env_prefix="AGENT_", env_file=".env", extra="ignore")


settings = Settings()
