# tests/test_config.py
import os
import pytest


def test_settings_loads_from_env(monkeypatch):
    monkeypatch.setenv("GITLAB_TOKEN", "test-token")
    monkeypatch.setenv("GITLAB_WEBHOOK_SECRET", "test-secret")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")

    # Re-import to pick up env vars
    from ai_review.config import Settings
    settings = Settings()

    assert settings.gitlab_token == "test-token"
    assert settings.gitlab_webhook_secret == "test-secret"
    assert settings.gemini_api_key == "test-gemini-key"


def test_settings_default_provider():
    from ai_review.config import Settings
    settings = Settings(
        gitlab_token="x",
        gitlab_webhook_secret="x",
        gemini_api_key="x",
    )
    assert settings.default_provider == "gemini"
    assert settings.default_language == "en"
