from typing import Any
from urllib.parse import quote
import httpx
from .base import GitPlatform


class GitLabClient(GitPlatform):
    def __init__(self, token: str, base_url: str = "https://gitlab.com"):
        self.token = token
        self.base_url = base_url.rstrip("/")
        self.api_url = f"{self.base_url}/api/v4"

    def _headers(self) -> dict[str, str]:
        return {"PRIVATE-TOKEN": self.token}

    async def get_mr_changes(self, project_id: int, mr_iid: int) -> dict[str, Any]:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.api_url}/projects/{project_id}/merge_requests/{mr_iid}/changes",
                headers=self._headers(),
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()

    async def get_file_content(self, project_id: int, file_path: str, ref: str) -> str:
        encoded_path = quote(file_path, safe="")
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.api_url}/projects/{project_id}/repository/files/{encoded_path}/raw",
                params={"ref": ref},
                headers=self._headers(),
                timeout=30.0,
            )
            response.raise_for_status()
            return response.text

    async def get_repo_config(self, project_id: int, ref: str) -> str | None:
        """Get .ai-review.yaml content, returns None if not found."""
        try:
            return await self.get_file_content(project_id, ".ai-review.yaml", ref)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    async def post_inline_comment(
        self,
        project_id: int,
        mr_iid: int,
        file_path: str,
        line: int,
        comment: str,
        diff_refs: dict[str, str],
    ) -> None:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.api_url}/projects/{project_id}/merge_requests/{mr_iid}/discussions",
                headers=self._headers(),
                json={
                    "body": comment,
                    "position": {
                        "position_type": "text",
                        "new_path": file_path,
                        "new_line": line,
                        "base_sha": diff_refs["base_sha"],
                        "head_sha": diff_refs["head_sha"],
                        "start_sha": diff_refs["start_sha"],
                    }
                },
                timeout=30.0,
            )
            response.raise_for_status()

    async def post_summary_comment(
        self,
        project_id: int,
        mr_iid: int,
        comment: str,
    ) -> None:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.api_url}/projects/{project_id}/merge_requests/{mr_iid}/notes",
                headers=self._headers(),
                json={"body": comment},
                timeout=30.0,
            )
            response.raise_for_status()
