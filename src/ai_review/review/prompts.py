from ai_review.models.config import RepoConfig


SYSTEM_PROMPT = """You are an AI code reviewer. Analyze the code changes for the following categories: {checks}.

Respond in language: {language}.

Return ONLY valid JSON in this exact format:
{{
  "comments": [
    {{
      "line": <line number in the NEW file>,
      "severity": "critical|high|medium|low",
      "category": "bugs|security|performance|readability|best-practices|logic",
      "comment": "<your comment text>"
    }}
  ],
  "summary": "<2-3 sentence summary of the changes>"
}}

Important:
- Only comment on the changed lines (from the diff)
- Use EXACTLY the line numbers shown in the numbered file listing (the number before the | symbol)
- Be specific and actionable in your comments
- If the code looks good, return empty comments array

Task description review (category: "logic"):
If a task description is provided, you MUST carefully compare the implementation against the requirements.
Check for:
- Missing requirements: features described but not implemented
- Incorrect logic: code that does something different from what the description specifies
- Edge cases: scenarios mentioned in the description but not handled in code
- Contradictions: code behavior that conflicts with the described requirements
Use category "logic" and severity "high" or "critical" for these issues."""


USER_PROMPT = """File: {file_path}
{description_block}
Full file with line numbers:
```
{numbered_content}
```

Changes (diff):
```diff
{diff_content}
```

Review the changes and respond with JSON. Use the line numbers shown above."""


def _add_line_numbers(content: str) -> str:
    """Add line numbers to file content for accurate LLM referencing."""
    lines = content.split("\n")
    width = len(str(len(lines)))
    return "\n".join(f"{i + 1:>{width}}| {line}" for i, line in enumerate(lines))


def build_review_prompt(
    file_path: str,
    file_content: str,
    diff_content: str,
    config: RepoConfig,
    mr_description: str = "",
) -> str:
    """Build the complete prompt for code review."""
    checks_str = ", ".join(check.value for check in config.checks)

    system = SYSTEM_PROMPT.format(
        checks=checks_str,
        language=config.language,
    )

    description_block = ""
    if mr_description and mr_description.strip():
        description_block = f"\n\nTask description (from MR):\n```\n{mr_description.strip()}\n```\n"

    user = USER_PROMPT.format(
        file_path=file_path,
        numbered_content=_add_line_numbers(file_content),
        diff_content=diff_content,
        description_block=description_block,
    )

    return f"{system}\n\n{user}"
