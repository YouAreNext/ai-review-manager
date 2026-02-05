# src/ai_review/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    # GitLab
    gitlab_url: str = "https://gitlab.com"
    gitlab_token: str
    gitlab_webhook_secret: str

    # LLM Providers
    gemini_api_key: str | None = None
    yandex_api_key: str | None = None
    yandex_folder_id: str | None = None

    # Defaults
    default_provider: str = "gemini"
    default_language: str = "en"
    reviewer_name: str = "AI Review"
    log_dir: str | None = None
    log_level: str = "INFO"
