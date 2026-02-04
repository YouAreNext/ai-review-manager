# tests/integration/test_api_review.py
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, AsyncMock
from ai_review.main import app, parse_gitlab_mr_url
from ai_review.review.engine import EngineReviewResult


def test_parse_gitlab_mr_url_valid():
    path, mr_iid = parse_gitlab_mr_url("https://gitlab.com/user/repo/-/merge_requests/123")
    assert path == "user/repo"
    assert mr_iid == 123


def test_parse_gitlab_mr_url_nested_path():
    path, mr_iid = parse_gitlab_mr_url("https://gitlab.com/group/subgroup/repo/-/merge_requests/45")
    assert path == "group/subgroup/repo"
    assert mr_iid == 45


def test_parse_gitlab_mr_url_invalid():
    with pytest.raises(ValueError):
        parse_gitlab_mr_url("https://github.com/user/repo/pull/123")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_trigger_review_by_project_id():
    transport = ASGITransport(app=app)

    mock_gitlab = AsyncMock()
    mock_gitlab.get_mr_info.return_value = {"source_branch": "feature-branch"}
    mock_gitlab.get_mr_changes.return_value = {
        "changes": [],
        "diff_refs": {"base_sha": "a", "head_sha": "b", "start_sha": "a"},
    }
    mock_gitlab.get_repo_config.return_value = None

    mock_provider = AsyncMock()
    mock_provider.review.return_value = AsyncMock(
        comments=[],
        summary="No issues found",
    )

    with patch("ai_review.main.get_settings") as mock_settings, \
         patch("ai_review.main.GitLabClient") as mock_gitlab_cls, \
         patch("ai_review.main.get_provider") as mock_get_provider:

        mock_settings.return_value.gitlab_token = "test-token"
        mock_gitlab_cls.return_value = mock_gitlab
        mock_get_provider.return_value = mock_provider

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/review",
                json={"project_id": 123, "mr_iid": 45},
            )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["project_id"] == 123
    assert data["mr_iid"] == 45


@pytest.mark.integration
@pytest.mark.asyncio
async def test_trigger_review_by_url():
    transport = ASGITransport(app=app)

    mock_gitlab = AsyncMock()
    mock_gitlab.get_project_by_path.return_value = {"id": 456}
    mock_gitlab.get_mr_info.return_value = {"source_branch": "feature-branch"}
    mock_gitlab.get_mr_changes.return_value = {
        "changes": [],
        "diff_refs": {"base_sha": "a", "head_sha": "b", "start_sha": "a"},
    }
    mock_gitlab.get_repo_config.return_value = None

    mock_provider = AsyncMock()
    mock_provider.review.return_value = AsyncMock(
        comments=[],
        summary="LGTM",
    )

    with patch("ai_review.main.get_settings") as mock_settings, \
         patch("ai_review.main.GitLabClient") as mock_gitlab_cls, \
         patch("ai_review.main.get_provider") as mock_get_provider:

        mock_settings.return_value.gitlab_token = "test-token"
        mock_gitlab_cls.return_value = mock_gitlab
        mock_get_provider.return_value = mock_provider

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/review",
                json={"url": "https://gitlab.com/user/repo/-/merge_requests/10"},
            )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["project_id"] == 456
    assert data["mr_iid"] == 10

    mock_gitlab.get_project_by_path.assert_called_once_with("user/repo")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_trigger_review_validation_error():
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/review",
            json={},  # No url or project_id+mr_iid
        )

    assert response.status_code == 422  # Validation error


@pytest.mark.integration
@pytest.mark.asyncio
async def test_trigger_review_no_provider():
    transport = ASGITransport(app=app)

    mock_gitlab = AsyncMock()
    mock_gitlab.get_mr_info.return_value = {"source_branch": "feature"}

    with patch("ai_review.main.get_settings") as mock_settings, \
         patch("ai_review.main.GitLabClient") as mock_gitlab_cls, \
         patch("ai_review.main.get_provider") as mock_get_provider:

        mock_settings.return_value.gitlab_token = "test-token"
        mock_gitlab_cls.return_value = mock_gitlab
        mock_get_provider.return_value = None  # No provider

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/review",
                json={"project_id": 123, "mr_iid": 45},
            )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "error"
    assert "No LLM provider" in data["error"]
