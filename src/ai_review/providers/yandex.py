# src/ai_review/providers/yandex.py
import json
import re
import httpx
from .base import LLMProvider
from ai_review.models.review import ReviewResult


class YandexProvider(LLMProvider):
    API_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

    def __init__(self, api_key: str, folder_id: str):
        self.api_key = api_key
        self.folder_id = folder_id

    async def review(self, prompt: str) -> ReviewResult:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.API_URL,
                headers={
                    "Authorization": f"Api-Key {self.api_key}",
                    "x-folder-id": self.folder_id,
                },
                json={
                    "modelUri": f"gpt://{self.folder_id}/yandexgpt/latest",
                    "completionOptions": {
                        "temperature": 0.3,
                        "maxTokens": 8000,
                    },
                    "messages": [
                        {"role": "user", "text": prompt}
                    ]
                },
                timeout=60.0
            )
            response.raise_for_status()

        data = response.json()
        text = data["result"]["alternatives"][0]["message"]["text"]

        # Extract JSON from response
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)

        result_data = json.loads(text)
        return ReviewResult(**result_data)
