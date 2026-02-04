import pytest
from ai_review.review.prompts import build_review_prompt
from ai_review.models.config import RepoConfig, CheckType


def test_build_prompt_includes_file_content():
    prompt = build_review_prompt(
        file_path="src/main.py",
        file_content="def hello():\n    print('hello')",
        diff_content="+ print('world')",
        config=RepoConfig(),
    )

    assert "src/main.py" in prompt
    assert "def hello():" in prompt
    assert "+ print('world')" in prompt


def test_build_prompt_includes_language():
    config = RepoConfig(language="ru")
    prompt = build_review_prompt(
        file_path="main.py",
        file_content="x = 1",
        diff_content="+x = 1",
        config=config,
    )

    assert "ru" in prompt.lower() or "русск" in prompt.lower()


def test_build_prompt_includes_checks():
    config = RepoConfig(checks=[CheckType.SECURITY, CheckType.BUGS])
    prompt = build_review_prompt(
        file_path="main.py",
        file_content="x = 1",
        diff_content="+x = 1",
        config=config,
    )

    assert "security" in prompt.lower()
    assert "bugs" in prompt.lower()
