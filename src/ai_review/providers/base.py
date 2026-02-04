# src/ai_review/providers/base.py
from abc import ABC, abstractmethod
from ai_review.models.review import ReviewResult


class LLMProvider(ABC):
    @abstractmethod
    async def review(self, prompt: str) -> ReviewResult:
        """Send prompt to LLM and return parsed review result."""
        pass
