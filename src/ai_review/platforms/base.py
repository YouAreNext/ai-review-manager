from abc import ABC, abstractmethod
from typing import Any


class GitPlatform(ABC):
    @abstractmethod
    async def get_mr_changes(self, project_id: int, mr_iid: int) -> dict[str, Any]:
        pass

    @abstractmethod
    async def get_file_content(self, project_id: int, file_path: str, ref: str) -> str:
        pass

    @abstractmethod
    async def post_inline_comment(
        self,
        project_id: int,
        mr_iid: int,
        file_path: str,
        line: int,
        comment: str,
        diff_refs: dict[str, str],
    ) -> None:
        pass

    @abstractmethod
    async def post_summary_comment(
        self,
        project_id: int,
        mr_iid: int,
        comment: str,
    ) -> None:
        pass
