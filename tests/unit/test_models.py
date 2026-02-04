# tests/test_models.py
import pytest
from ai_review.models.config import RepoConfig, CheckType
from ai_review.models.review import ReviewComment, ReviewResult, Severity


def test_repo_config_defaults():
    config = RepoConfig()
    assert config.language == "en"
    assert config.provider == "gemini"
    assert config.auto_review is True
    assert CheckType.BUGS in config.checks
    assert CheckType.SECURITY in config.checks


def test_repo_config_from_yaml():
    yaml_content = """
language: ru
provider: yandex
checks:
  - bugs
  - security
auto_review: false
min_severity: medium
exclude:
  - "*.generated.ts"
"""
    import yaml
    data = yaml.safe_load(yaml_content)
    config = RepoConfig(**data)

    assert config.language == "ru"
    assert config.provider == "yandex"
    assert config.auto_review is False
    assert config.min_severity == Severity.MEDIUM


def test_review_comment_model():
    comment = ReviewComment(
        line=42,
        severity=Severity.HIGH,
        category=CheckType.SECURITY,
        comment="SQL injection vulnerability"
    )
    assert comment.line == 42
    assert comment.severity == Severity.HIGH


def test_review_result_model():
    result = ReviewResult(
        comments=[
            ReviewComment(
                line=10,
                severity=Severity.LOW,
                category=CheckType.READABILITY,
                comment="Consider renaming variable"
            )
        ],
        summary="Minor readability improvements suggested"
    )
    assert len(result.comments) == 1
    assert result.summary is not None
