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
    model_config = SettingsConfigDict(env_prefix="AGENT_", env_file=".env", extra="ignore")


settings = Settings()
