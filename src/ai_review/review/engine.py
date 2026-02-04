# src/ai_review/review/engine.py
import yaml
import logging
from ai_review.platforms.gitlab import GitLabClient
from ai_review.providers.base import LLMProvider
from ai_review.models.config import RepoConfig, Severity
from ai_review.models.review import ReviewComment
from .parser import parse_diff
from .prompts import build_review_prompt


logger = logging.getLogger(__name__)

SEVERITY_ORDER = {
    Severity.LOW: 0,
    Severity.MEDIUM: 1,
    Severity.HIGH: 2,
    Severity.CRITICAL: 3,
}


class ReviewEngine:
    def __init__(self, gitlab: GitLabClient, provider: LLMProvider):
        self.gitlab = gitlab
        self.provider = provider

    async def review_mr(
        self,
        project_id: int,
        mr_iid: int,
        source_branch: str,
    ) -> None:
        """Run AI review on a merge request."""
        # Get MR changes
        mr_data = await self.gitlab.get_mr_changes(project_id, mr_iid)
        changes = mr_data["changes"]
        diff_refs = mr_data["diff_refs"]

        # Load repo config
        config = await self._load_config(project_id, source_branch)

        all_comments: list[tuple[str, ReviewComment]] = []
        summaries: list[str] = []

        for change in changes:
            file_path = change["new_path"]

            # Skip excluded files
            if self._is_excluded(file_path, config.exclude):
                continue

            # Get file content for context
            try:
                file_content = await self.gitlab.get_file_content(
                    project_id, file_path, source_branch
                )
            except Exception as e:
                logger.warning(f"Could not get file content for {file_path}: {e}")
                file_content = ""

            # Build prompt and call LLM
            prompt = build_review_prompt(
                file_path=file_path,
                file_content=file_content,
                diff_content=change["diff"],
                config=config,
            )

            try:
                result = await self.provider.review(prompt)
            except Exception as e:
                logger.error(f"LLM review failed for {file_path}: {e}")
                continue

            # Collect comments
            for comment in result.comments:
                if self._passes_severity_threshold(comment.severity, config.min_severity):
                    all_comments.append((file_path, comment))

            if result.summary:
                summaries.append(f"**{file_path}**: {result.summary}")

        # Post inline comments
        for file_path, comment in all_comments:
            formatted = self._format_comment(comment)
            try:
                await self.gitlab.post_inline_comment(
                    project_id=project_id,
                    mr_iid=mr_iid,
                    file_path=file_path,
                    line=comment.line,
                    comment=formatted,
                    diff_refs=diff_refs,
                )
            except Exception as e:
                logger.error(f"Failed to post comment: {e}")

        # Post summary
        if summaries:
            summary_text = "## AI Review Summary\n\n" + "\n\n".join(summaries)
            await self.gitlab.post_summary_comment(project_id, mr_iid, summary_text)

    async def _load_config(self, project_id: int, ref: str) -> RepoConfig:
        """Load .ai-review.yaml from repo or use defaults."""
        yaml_content = await self.gitlab.get_repo_config(project_id, ref)
        if yaml_content is None:
            return RepoConfig()

        try:
            data = yaml.safe_load(yaml_content)
            return RepoConfig(**data)
        except Exception as e:
            logger.warning(f"Invalid .ai-review.yaml: {e}")
            return RepoConfig()

    def _is_excluded(self, file_path: str, patterns: list[str]) -> bool:
        """Check if file matches any exclude pattern."""
        import fnmatch
        for pattern in patterns:
            if fnmatch.fnmatch(file_path, pattern):
                return True
        return False

    def _passes_severity_threshold(self, severity: Severity, threshold: Severity) -> bool:
        """Check if comment severity meets threshold."""
        return SEVERITY_ORDER[severity] >= SEVERITY_ORDER[threshold]

    def _format_comment(self, comment: ReviewComment) -> str:
        """Format comment for GitLab."""
        emoji = {
            Severity.CRITICAL: "🚨",
            Severity.HIGH: "⚠️",
            Severity.MEDIUM: "💡",
            Severity.LOW: "ℹ️",
        }
        return f"{emoji[comment.severity]} **{comment.category.value}** | `{comment.severity.value}`\n\n{comment.comment}"
