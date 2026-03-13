from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """
    Central configuration using pydantic-settings.
    Values are read (in order of priority):
    1. Environment variables (highest priority)
    2. .env file in project root
    3. Default values defined here
    """

    model_config = SettingsConfigDict(
        env_file=".env",              # auto-load .env
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",               # ignore unknown env vars
    )

    # LLM settings
    default_model: str = "llama3.2:3b"
    default_temperature: float = 0.7
    default_max_tokens: int = 512

    # Logging & observability
    log_level: str = "INFO"           # DEBUG, INFO, WARNING, ERROR, CRITICAL
    metrics_file: str = "logs/llm-calls.jsonl"
    metrics_max_bytes: int = 10 * 1024 * 1024  # 10 MB
    metrics_backup_count: int = 5

    # Future placeholders (commented for now)
    # openai_api_key: Optional[str] = None
    # aws_region: Optional[str] = "us-east-1"

    @property
    def log_level_int(self) -> int:
        """Convert string log level to logging integer constant."""
        levels = {
            "DEBUG": 10,
            "INFO": 20,
            "WARNING": 30,
            "ERROR": 40,
            "CRITICAL": 50,
        }
        return levels.get(self.log_level.upper(), 20)  # default INFO


# Singleton instance — import and use this everywhere
settings = Settings()