# tests/integration/test_providers.py
import pytest
from ai_review.providers.gemini import GeminiProvider


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gemini_provider_builds_request(httpx_mock):
    httpx_mock.add_response(
        url="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=test-key",
        json={
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": '{"comments": [], "summary": "LGTM"}'
                    }]
                }
            }]
        }
    )

    provider = GeminiProvider(api_key="test-key")
    result = await provider.review("Review this code: print('hello')")

    assert result.summary == "LGTM"
    assert result.comments == []
