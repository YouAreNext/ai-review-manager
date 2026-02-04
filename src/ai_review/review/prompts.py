from ai_review.models.config import RepoConfig


SYSTEM_PROMPT = """You are an AI code reviewer. Analyze the code changes for the following categories: {checks}.

Respond in language: {language}.

Return ONLY valid JSON in this exact format:
{{
  "comments": [
    {{
      "line": <line number in the NEW file>,
      "severity": "critical|high|medium|low",
      "category": "bugs|security|performance|readability|best-practices",
      "comment": "<your comment text>"
    }}
  ],
  "summary": "<2-3 sentence summary of the changes>"
}}

Important:
- Only comment on the changed lines (from the diff)
- Line numbers must match the NEW file, not the old one
- Be specific and actionable in your comments
- If the code looks good, return empty comments array"""


USER_PROMPT = """File: {file_path}

Full file for context:
```
{file_content}
```

Changes (diff):
```diff
{diff_content}
```

Review the changes and respond with JSON."""


def build_review_prompt(
    file_path: str,
    file_content: str,
    diff_content: str,
    config: RepoConfig,
) -> str:
    """Build the complete prompt for code review."""
    checks_str = ", ".join(check.value for check in config.checks)

    system = SYSTEM_PROMPT.format(
        checks=checks_str,
        language=config.language,
    )

    user = USER_PROMPT.format(
        file_path=file_path,
        file_content=file_content,
        diff_content=diff_content,
    )

    return f"{system}\n\n{user}"
