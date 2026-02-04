# tests/test_engine.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from ai_review.review.engine import ReviewEngine
from ai_review.models.config import RepoConfig, CheckType
from ai_review.models.review import ReviewResult, ReviewComment, Severity


@pytest.fixture
def mock_gitlab():
    client = AsyncMock()
    client.get_mr_changes.return_value = {
        "changes": [{
            "old_path": "main.py",
            "new_path": "main.py",
            "diff": "@@ -1 +1,2 @@\n+print('hello')",
        }],
        "diff_refs": {
            "base_sha": "abc",
            "head_sha": "def",
            "start_sha": "abc",
        }
    }
    client.get_file_content.return_value = "print('hello')"
    client.get_repo_config.return_value = None
    return client


@pytest.fixture
def mock_provider():
    provider = AsyncMock()
    provider.review.return_value = ReviewResult(
        comments=[
            ReviewComment(
                line=2,
                severity=Severity.LOW,
                category=CheckType.READABILITY,
                comment="Consider using logging"
            )
        ],
        summary="Minor suggestion for improvement"
    )
    return provider


@pytest.mark.asyncio
async def test_engine_runs_review(mock_gitlab, mock_provider):
    engine = ReviewEngine(
        gitlab=mock_gitlab,
        provider=mock_provider,
    )

    await engine.review_mr(project_id=123, mr_iid=45, source_branch="feature")

    mock_gitlab.get_mr_changes.assert_called_once()
    mock_provider.review.assert_called_once()
    mock_gitlab.post_inline_comment.assert_called_once()
    mock_gitlab.post_summary_comment.assert_called_once()


@pytest.mark.asyncio
async def test_engine_filters_by_severity(mock_gitlab, mock_provider):
    mock_gitlab.get_repo_config.return_value = "min_severity: high"

    engine = ReviewEngine(
        gitlab=mock_gitlab,
        provider=mock_provider,
    )

    await engine.review_mr(project_id=123, mr_iid=45, source_branch="feature")

    # Low severity comment should be filtered out
    mock_gitlab.post_inline_comment.assert_not_called()
