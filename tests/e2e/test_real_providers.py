# tests/e2e/test_real_providers.py
"""
End-to-end tests for LLM providers with real API calls.

These tests require valid API credentials set in environment variables:
- GEMINI_API_KEY: Google Gemini API key
- YANDEX_API_KEY: Yandex GPT API key
- YANDEX_FOLDER_ID: Yandex Cloud folder ID

Run with: pytest tests/e2e/ -m e2e -v
"""
import os
import pytest
from ai_review.providers.gemini import GeminiProvider
from ai_review.providers.yandex import YandexProvider
from ai_review.models.config import RepoConfig
from ai_review.review.prompts import build_review_prompt


SIMPLE_CODE = "def add(a, b):\n    return a + b"
SIMPLE_DIFF = "+def add(a, b):\n+    return a + b"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_gemini_real_review():
    """Test Gemini provider with real API call."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY not set")

    provider = GeminiProvider(api_key=api_key)
    prompt = build_review_prompt(
        file_path="math.py",
        file_content=SIMPLE_CODE,
        diff_content=SIMPLE_DIFF,
        config=RepoConfig(language="en"),
    )

    result = await provider.review(prompt)

    assert result is not None
    assert result.summary is not None
    assert isinstance(result.comments, list)
    print(f"\nGemini summary: {result.summary}")
    print(f"Gemini comments: {len(result.comments)}")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_yandex_real_review():
    """Test Yandex provider with real API call."""
    api_key = os.environ.get("YANDEX_API_KEY")
    folder_id = os.environ.get("YANDEX_FOLDER_ID")
    if not api_key or not folder_id:
        pytest.skip("YANDEX_API_KEY or YANDEX_FOLDER_ID not set")

    provider = YandexProvider(api_key=api_key, folder_id=folder_id)
    prompt = build_review_prompt(
        file_path="math.py",
        file_content=SIMPLE_CODE,
        diff_content=SIMPLE_DIFF,
        config=RepoConfig(language="ru"),
    )

    result = await provider.review(prompt)

    assert result is not None
    assert result.summary is not None
    assert isinstance(result.comments, list)
    print(f"\nYandex summary: {result.summary}")
    print(f"Yandex comments: {len(result.comments)}")
