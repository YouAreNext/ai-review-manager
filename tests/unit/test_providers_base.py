# tests/unit/test_providers_base.py
import pytest
from ai_review.providers.base import LLMProvider


@pytest.mark.unit
def test_provider_is_abstract():
    with pytest.raises(TypeError):
        LLMProvider()
