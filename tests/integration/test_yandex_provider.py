# tests/integration/test_yandex_provider.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from ai_review.providers.yandex import YandexProvider


def _mock_completion(text: str):
    """Create a mock ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = text
    completion = MagicMock()
    completion.choices = [choice]
    return completion


@pytest.mark.integration
@pytest.mark.asyncio
async def test_yandex_provider_returns_review():
    provider = YandexProvider(api_key="test-key", folder_id="test-folder")
    provider.client = AsyncMock()
    provider.client.chat.completions.create = AsyncMock(
        return_value=_mock_completion('{"comments": [], "summary": "Code looks good"}')
    )

    result = await provider.review("Review this code")

    assert result.summary == "Code looks good"
    assert result.comments == []

    provider.client.chat.completions.create.assert_called_once()
    call_kwargs = provider.client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt://test-folder/yandexgpt/latest"
    assert call_kwargs["messages"] == [{"role": "user", "content": "Review this code"}]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_yandex_provider_extracts_json_from_markdown():
    provider = YandexProvider(api_key="test-key", folder_id="test-folder")
    provider.client = AsyncMock()
    provider.client.chat.completions.create = AsyncMock(
        return_value=_mock_completion(
            'Here is my review:\n```json\n{"comments": [], "summary": "All good"}\n```'
        )
    )

    result = await provider.review("Review this code")

    assert result.summary == "All good"
    assert result.comments == []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_yandex_provider_with_comments():
    provider = YandexProvider(api_key="test-key", folder_id="test-folder")
    provider.client = AsyncMock()
    provider.client.chat.completions.create = AsyncMock(
        return_value=_mock_completion(
            '{"comments": [{"line": 10, "severity": "high", "category": "bugs", "comment": "Null check missing"}], "summary": "Found a bug"}'
        )
    )

    result = await provider.review("Review this code")

    assert result.summary == "Found a bug"
    assert len(result.comments) == 1
    assert result.comments[0].line == 10
    assert result.comments[0].comment == "Null check missing"
