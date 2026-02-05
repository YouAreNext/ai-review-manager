# src/ai_review/providers/gemini.py
import json
import re
import httpx
from .base import LLMProvider
from ai_review.models.review import ReviewResult


class GeminiProvider(LLMProvider):
    API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent"

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def review(self, prompt: str) -> ReviewResult:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.API_URL}?key={self.api_key}",
                json={
                    "contents": [{
                        "parts": [{"text": prompt}]
                    }]
                },
                timeout=60.0
            )
            response.raise_for_status()

        data = response.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]

        # Extract JSON from response (may be wrapped in ```json or just ```)
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)

        result_data = json.loads(text)
        return ReviewResult(**result_data)
