# tests/integration/test_webhook.py
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch
from ai_review.main import app


@pytest.mark.integration
@pytest.mark.asyncio
async def test_webhook_rejects_invalid_token():
    transport = ASGITransport(app=app)

    with patch("ai_review.main.get_settings") as mock_settings:
        mock_settings.return_value.gitlab_webhook_secret = "correct-secret"
        mock_settings.return_value.gitlab_token = "test-token"

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/webhook/gitlab",
                headers={"X-Gitlab-Token": "wrong-token"},
                json={"object_kind": "merge_request"}
            )

    assert response.status_code == 401


@pytest.mark.integration
@pytest.mark.asyncio
async def test_webhook_accepts_mr_event():
    transport = ASGITransport(app=app)

    with patch("ai_review.main.get_settings") as mock_settings:
        mock_settings.return_value.gitlab_webhook_secret = "test-secret"
        mock_settings.return_value.gitlab_token = "test-token"
        mock_settings.return_value.gemini_api_key = "test-key"
        mock_settings.return_value.default_provider = "gemini"

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/webhook/gitlab",
                headers={"X-Gitlab-Token": "test-secret"},
                json={
                    "object_kind": "merge_request",
                    "user": {"username": "test"},
                    "project": {"id": 123, "path_with_namespace": "test/repo", "web_url": "https://gitlab.com/test/repo"},
                    "object_attributes": {
                        "iid": 1,
                        "title": "Test MR",
                        "source_branch": "feature",
                        "target_branch": "main",
                        "state": "opened",
                        "action": "open"
                    }
                }
            )

    assert response.status_code == 200
    assert response.json()["status"] == "accepted"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_webhook_triggers_on_review_command():
    transport = ASGITransport(app=app)

    with patch("ai_review.main.get_settings") as mock_settings:
        mock_settings.return_value.gitlab_webhook_secret = "test-secret"
        mock_settings.return_value.gitlab_token = "test-token"
        mock_settings.return_value.gemini_api_key = "test-key"
        mock_settings.return_value.default_provider = "gemini"

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/webhook/gitlab",
                headers={"X-Gitlab-Token": "test-secret"},
                json={
                    "object_kind": "note",
                    "user": {"username": "test"},
                    "project": {"id": 123, "path_with_namespace": "test/repo", "web_url": "https://gitlab.com/test/repo"},
                    "merge_request": {
                        "iid": 1,
                        "title": "Test MR",
                        "source_branch": "feature",
                        "target_branch": "main",
                        "state": "opened"
                    },
                    "object_attributes": {
                        "note": "/review",
                        "noteable_type": "MergeRequest"
                    }
                }
            )

    assert response.status_code == 200
    assert response.json()["status"] == "accepted"
