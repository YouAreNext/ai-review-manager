# tests/test_yandex_provider.py
import pytest
from ai_review.providers.yandex import YandexProvider


@pytest.mark.asyncio
async def test_yandex_provider_builds_request(httpx_mock):
    httpx_mock.add_response(
        url="https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
        json={
            "result": {
                "alternatives": [{
                    "message": {
                        "text": '{"comments": [], "summary": "Code looks good"}'
                    }
                }]
            }
        }
    )

    provider = YandexProvider(api_key="test-key", folder_id="test-folder")
    result = await provider.review("Review this code")

    assert result.summary == "Code looks good"
    assert result.comments == []
