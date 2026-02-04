# tests/integration/test_gitlab_client.py
import pytest
from ai_review.platforms.gitlab import GitLabClient


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_mr_changes(httpx_mock):
    httpx_mock.add_response(
        url="https://gitlab.com/api/v4/projects/123/merge_requests/45/changes",
        json={
            "changes": [
                {
                    "old_path": "src/main.py",
                    "new_path": "src/main.py",
                    "diff": "@@ -1 +1,2 @@\n+print('hello')",
                }
            ],
            "diff_refs": {
                "base_sha": "abc123",
                "head_sha": "def456",
                "start_sha": "abc123",
            }
        }
    )

    client = GitLabClient(token="test-token", base_url="https://gitlab.com")
    changes = await client.get_mr_changes(project_id=123, mr_iid=45)

    assert len(changes["changes"]) == 1
    assert changes["changes"][0]["new_path"] == "src/main.py"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_file_content(httpx_mock):
    httpx_mock.add_response(
        url="https://gitlab.com/api/v4/projects/123/repository/files/src%2Fmain.py/raw?ref=feature-branch",
        text="print('hello world')",
    )

    client = GitLabClient(token="test-token", base_url="https://gitlab.com")
    content = await client.get_file_content(
        project_id=123,
        file_path="src/main.py",
        ref="feature-branch"
    )

    assert content == "print('hello world')"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_post_inline_comment(httpx_mock):
    httpx_mock.add_response(
        url="https://gitlab.com/api/v4/projects/123/merge_requests/45/discussions",
        json={"id": "discussion-1"},
    )

    client = GitLabClient(token="test-token", base_url="https://gitlab.com")
    await client.post_inline_comment(
        project_id=123,
        mr_iid=45,
        file_path="src/main.py",
        line=10,
        comment="Consider error handling",
        diff_refs={"base_sha": "abc", "head_sha": "def", "start_sha": "abc"}
    )

    request = httpx_mock.get_request()
    assert request is not None
