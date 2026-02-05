# src/ai_review/providers/yandex.py
import json
import re
import logging
from openai import AsyncOpenAI
from .base import LLMProvider
from ai_review.models.review import ReviewResult


logger = logging.getLogger(__name__)


class YandexProvider(LLMProvider):
    BASE_URL = "https://llm.api.cloud.yandex.net/v1"

    def __init__(self, api_key: str, folder_id: str):
        self.api_key = api_key
        self.folder_id = folder_id
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.BASE_URL,
            default_headers={"x-folder-id": folder_id},
        )

    async def review(self, prompt: str) -> ReviewResult:
        response = await self.client.chat.completions.create(
            model=f"gpt://{self.folder_id}/yandexgpt/latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=8000,
        )

        text = response.choices[0].message.content or ""
        logger.info(f"Yandex response length: {len(text)} chars")
        logger.info(f"Yandex response first 500 chars: {text[:500]}")

        if not text.strip():
            raise ValueError(f"Yandex returned empty response. Full API response: {response}")

        # Extract JSON from response (may be wrapped in ```json or just ```)
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)

        result_data = json.loads(text)
        return ReviewResult(**result_data)
