# AI Review Manager MVP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a self-hosted FastAPI service that reviews GitLab Merge Requests using LLM (Gemini/Yandex).

**Architecture:** Webhook-driven FastAPI service. GitLab sends events on MR creation or `/review` comment. Service fetches diff, calls LLM, posts inline comments + summary back to MR.

**Tech Stack:** Python 3.11+, FastAPI, httpx, Pydantic, unidiff, pytest, Docker

---

## Task 1: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: `src/ai_review/__init__.py`
- Create: `src/ai_review/main.py`
- Create: `tests/__init__.py`
- Create: `tests/test_health.py`

**Step 1: Create pyproject.toml with dependencies**

```toml
[project]
name = "ai-review-manager"
version = "0.1.0"
description = "AI Code Reviewer for GitLab"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "httpx>=0.26.0",
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",
    "unidiff>=0.7.5",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.23.0",
    "pytest-httpx>=0.28.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

**Step 2: Create directory structure**

```bash
mkdir -p src/ai_review tests
touch src/ai_review/__init__.py tests/__init__.py
```

**Step 3: Write failing health endpoint test**

```python
# tests/test_health.py
import pytest
from httpx import AsyncClient, ASGITransport
from ai_review.main import app


@pytest.mark.asyncio
async def test_health_returns_ok():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "0.1.0"}
```

**Step 4: Run test to verify it fails**

```bash
pip install -e ".[dev]"
pytest tests/test_health.py -v
```

Expected: FAIL (no module `ai_review.main`)

**Step 5: Write minimal FastAPI app**

```python
# src/ai_review/main.py
from fastapi import FastAPI

app = FastAPI(title="AI Review Manager")


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}
```

**Step 6: Run test to verify it passes**

```bash
pytest tests/test_health.py -v
```

Expected: PASS

**Step 7: Commit**

```bash
git add pyproject.toml src/ tests/
git commit -m "feat: project setup with health endpoint"
```

---

## Task 2: Configuration (Settings)

**Files:**
- Create: `src/ai_review/config.py`
- Create: `tests/test_config.py`

**Step 1: Write failing config test**

```python
# tests/test_config.py
import os
import pytest


def test_settings_loads_from_env(monkeypatch):
    monkeypatch.setenv("GITLAB_TOKEN", "test-token")
    monkeypatch.setenv("GITLAB_WEBHOOK_SECRET", "test-secret")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")

    # Re-import to pick up env vars
    from ai_review.config import Settings
    settings = Settings()

    assert settings.gitlab_token == "test-token"
    assert settings.gitlab_webhook_secret == "test-secret"
    assert settings.gemini_api_key == "test-gemini-key"


def test_settings_default_provider():
    from ai_review.config import Settings
    settings = Settings(
        gitlab_token="x",
        gitlab_webhook_secret="x",
        gemini_api_key="x",
    )
    assert settings.default_provider == "gemini"
    assert settings.default_language == "en"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_config.py -v
```

Expected: FAIL (no module)

**Step 3: Write Settings class**

```python
# src/ai_review/config.py
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # GitLab
    gitlab_token: str
    gitlab_webhook_secret: str

    # LLM Providers
    gemini_api_key: str | None = None
    yandex_api_key: str | None = None
    yandex_folder_id: str | None = None

    # Defaults
    default_provider: str = "gemini"
    default_language: str = "en"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_config.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/ai_review/config.py tests/test_config.py
git commit -m "feat: add configuration with pydantic-settings"
```

---

## Task 3: Pydantic Models

**Files:**
- Create: `src/ai_review/models/__init__.py`
- Create: `src/ai_review/models/config.py`
- Create: `src/ai_review/models/review.py`
- Create: `src/ai_review/models/webhook.py`
- Create: `tests/test_models.py`

**Step 1: Write failing models test**

```python
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
```

**Step 2: Run test to verify it fails**

```bash
pip install pyyaml  # needed for test
pytest tests/test_models.py -v
```

Expected: FAIL (no module)

**Step 3: Create models**

```python
# src/ai_review/models/__init__.py
from .config import RepoConfig, CheckType
from .review import ReviewComment, ReviewResult, Severity
from .webhook import GitLabMREvent, GitLabNoteEvent

__all__ = [
    "RepoConfig",
    "CheckType",
    "ReviewComment",
    "ReviewResult",
    "Severity",
    "GitLabMREvent",
    "GitLabNoteEvent",
]
```

```python
# src/ai_review/models/config.py
from enum import Enum
from pydantic import BaseModel, Field


class CheckType(str, Enum):
    BUGS = "bugs"
    SECURITY = "security"
    PERFORMANCE = "performance"
    READABILITY = "readability"
    BEST_PRACTICES = "best-practices"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RepoConfig(BaseModel):
    language: str = "en"
    provider: str = "gemini"
    checks: list[CheckType] = Field(
        default_factory=lambda: [
            CheckType.BUGS,
            CheckType.SECURITY,
            CheckType.PERFORMANCE,
            CheckType.READABILITY,
            CheckType.BEST_PRACTICES,
        ]
    )
    exclude: list[str] = Field(default_factory=list)
    auto_review: bool = True
    min_severity: Severity = Severity.LOW
```

```python
# src/ai_review/models/review.py
from pydantic import BaseModel
from .config import CheckType, Severity


class ReviewComment(BaseModel):
    line: int
    severity: Severity
    category: CheckType
    comment: str


class ReviewResult(BaseModel):
    comments: list[ReviewComment]
    summary: str
```

```python
# src/ai_review/models/webhook.py
from pydantic import BaseModel


class GitLabUser(BaseModel):
    username: str
    name: str | None = None


class GitLabProject(BaseModel):
    id: int
    path_with_namespace: str
    web_url: str


class GitLabMergeRequest(BaseModel):
    iid: int
    title: str
    source_branch: str
    target_branch: str
    state: str
    action: str | None = None


class GitLabMREvent(BaseModel):
    object_kind: str  # "merge_request"
    user: GitLabUser
    project: GitLabProject
    object_attributes: GitLabMergeRequest


class GitLabNote(BaseModel):
    note: str
    noteable_type: str


class GitLabNoteEvent(BaseModel):
    object_kind: str  # "note"
    user: GitLabUser
    project: GitLabProject
    merge_request: GitLabMergeRequest | None = None
    object_attributes: GitLabNote
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_models.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/ai_review/models/ tests/test_models.py
git commit -m "feat: add pydantic models for config, review, and webhooks"
```

---

## Task 4: LLM Provider Abstraction

**Files:**
- Create: `src/ai_review/providers/__init__.py`
- Create: `src/ai_review/providers/base.py`
- Create: `src/ai_review/providers/gemini.py`
- Create: `tests/test_providers.py`

**Step 1: Write failing provider test**

```python
# tests/test_providers.py
import pytest
from ai_review.providers.base import LLMProvider
from ai_review.providers.gemini import GeminiProvider


def test_provider_is_abstract():
    with pytest.raises(TypeError):
        LLMProvider()


@pytest.mark.asyncio
async def test_gemini_provider_builds_request(httpx_mock):
    httpx_mock.add_response(
        url="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=test-key",
        json={
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": '{"comments": [], "summary": "LGTM"}'
                    }]
                }
            }]
        }
    )

    provider = GeminiProvider(api_key="test-key")
    result = await provider.review("Review this code: print('hello')")

    assert result.summary == "LGTM"
    assert result.comments == []
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_providers.py -v
```

Expected: FAIL (no module)

**Step 3: Create provider abstraction**

```python
# src/ai_review/providers/__init__.py
from .base import LLMProvider
from .gemini import GeminiProvider

__all__ = ["LLMProvider", "GeminiProvider"]
```

```python
# src/ai_review/providers/base.py
from abc import ABC, abstractmethod
from ai_review.models.review import ReviewResult


class LLMProvider(ABC):
    @abstractmethod
    async def review(self, prompt: str) -> ReviewResult:
        """Send prompt to LLM and return parsed review result."""
        pass
```

```python
# src/ai_review/providers/gemini.py
import json
import re
import httpx
from .base import LLMProvider
from ai_review.models.review import ReviewResult


class GeminiProvider(LLMProvider):
    API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def review(self, prompt: str) -> ReviewResult:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.API_URL}?key={self.api_key}",
                json={
                    "contents": [{
                        "parts": [{"text": prompt}]
                    }]
                },
                timeout=60.0
            )
            response.raise_for_status()

        data = response.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]

        # Extract JSON from response (may be wrapped in ```json)
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)

        result_data = json.loads(text)
        return ReviewResult(**result_data)
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_providers.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/ai_review/providers/ tests/test_providers.py
git commit -m "feat: add LLM provider abstraction with Gemini implementation"
```

---

## Task 5: Yandex GPT Provider

**Files:**
- Create: `src/ai_review/providers/yandex.py`
- Modify: `src/ai_review/providers/__init__.py`
- Create: `tests/test_yandex_provider.py`

**Step 1: Write failing Yandex provider test**

```python
# tests/test_yandex_provider.py
import pytest
from ai_review.providers.yandex import YandexProvider


@pytest.mark.asyncio
async def test_yandex_provider_builds_request(httpx_mock):
    httpx_mock.add_response(
        url="https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
        json={
            "result": {
                "alternatives": [{
                    "message": {
                        "text": '{"comments": [], "summary": "Code looks good"}'
                    }
                }]
            }
        }
    )

    provider = YandexProvider(api_key="test-key", folder_id="test-folder")
    result = await provider.review("Review this code")

    assert result.summary == "Code looks good"
    assert result.comments == []
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_yandex_provider.py -v
```

Expected: FAIL (no module)

**Step 3: Create Yandex provider**

```python
# src/ai_review/providers/yandex.py
import json
import re
import httpx
from .base import LLMProvider
from ai_review.models.review import ReviewResult


class YandexProvider(LLMProvider):
    API_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

    def __init__(self, api_key: str, folder_id: str):
        self.api_key = api_key
        self.folder_id = folder_id

    async def review(self, prompt: str) -> ReviewResult:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.API_URL,
                headers={
                    "Authorization": f"Api-Key {self.api_key}",
                    "x-folder-id": self.folder_id,
                },
                json={
                    "modelUri": f"gpt://{self.folder_id}/yandexgpt/latest",
                    "completionOptions": {
                        "temperature": 0.3,
                        "maxTokens": 8000,
                    },
                    "messages": [
                        {"role": "user", "text": prompt}
                    ]
                },
                timeout=60.0
            )
            response.raise_for_status()

        data = response.json()
        text = data["result"]["alternatives"][0]["message"]["text"]

        # Extract JSON from response
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)

        result_data = json.loads(text)
        return ReviewResult(**result_data)
```

**Step 4: Update providers __init__.py**

```python
# src/ai_review/providers/__init__.py
from .base import LLMProvider
from .gemini import GeminiProvider
from .yandex import YandexProvider

__all__ = ["LLMProvider", "GeminiProvider", "YandexProvider"]
```

**Step 5: Run test to verify it passes**

```bash
pytest tests/test_yandex_provider.py -v
```

Expected: PASS

**Step 6: Commit**

```bash
git add src/ai_review/providers/ tests/test_yandex_provider.py
git commit -m "feat: add YandexGPT provider"
```

---

## Task 6: Diff Parser

**Files:**
- Create: `src/ai_review/review/__init__.py`
- Create: `src/ai_review/review/parser.py`
- Create: `tests/test_parser.py`

**Step 1: Write failing parser test**

```python
# tests/test_parser.py
import pytest
from ai_review.review.parser import parse_diff, DiffFile


SAMPLE_DIFF = """--- a/src/main.py
+++ b/src/main.py
@@ -10,6 +10,8 @@ def hello():
     print("hello")
+    print("world")
+    return True

 def goodbye():
     pass
"""


def test_parse_diff_extracts_files():
    files = parse_diff(SAMPLE_DIFF)

    assert len(files) == 1
    assert files[0].path == "src/main.py"
    assert files[0].is_new is False


def test_parse_diff_extracts_added_lines():
    files = parse_diff(SAMPLE_DIFF)

    added_lines = files[0].added_lines
    assert 12 in added_lines  # print("world")
    assert 13 in added_lines  # return True


def test_parse_diff_new_file():
    diff = """--- /dev/null
+++ b/new_file.py
@@ -0,0 +1,3 @@
+def new_func():
+    pass
+
"""
    files = parse_diff(diff)

    assert len(files) == 1
    assert files[0].path == "new_file.py"
    assert files[0].is_new is True
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_parser.py -v
```

Expected: FAIL (no module)

**Step 3: Create parser**

```python
# src/ai_review/review/__init__.py
from .parser import parse_diff, DiffFile

__all__ = ["parse_diff", "DiffFile"]
```

```python
# src/ai_review/review/parser.py
from dataclasses import dataclass, field
from unidiff import PatchSet


@dataclass
class DiffFile:
    path: str
    diff: str
    is_new: bool
    is_deleted: bool
    added_lines: list[int] = field(default_factory=list)


def parse_diff(diff_text: str) -> list[DiffFile]:
    """Parse unified diff and extract file information."""
    patch = PatchSet(diff_text)
    files = []

    for patched_file in patch:
        added_lines = []

        for hunk in patched_file:
            for line in hunk:
                if line.is_added and line.target_line_no is not None:
                    added_lines.append(line.target_line_no)

        files.append(DiffFile(
            path=patched_file.path,
            diff=str(patched_file),
            is_new=patched_file.is_added_file,
            is_deleted=patched_file.is_removed_file,
            added_lines=added_lines,
        ))

    return files
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_parser.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/ai_review/review/ tests/test_parser.py
git commit -m "feat: add diff parser using unidiff"
```

---

## Task 7: Prompt Builder

**Files:**
- Create: `src/ai_review/review/prompts.py`
- Modify: `src/ai_review/review/__init__.py`
- Create: `tests/test_prompts.py`

**Step 1: Write failing prompts test**

```python
# tests/test_prompts.py
import pytest
from ai_review.review.prompts import build_review_prompt
from ai_review.models.config import RepoConfig, CheckType


def test_build_prompt_includes_file_content():
    prompt = build_review_prompt(
        file_path="src/main.py",
        file_content="def hello():\n    print('hello')",
        diff_content="+ print('world')",
        config=RepoConfig(),
    )

    assert "src/main.py" in prompt
    assert "def hello():" in prompt
    assert "+ print('world')" in prompt


def test_build_prompt_includes_language():
    config = RepoConfig(language="ru")
    prompt = build_review_prompt(
        file_path="main.py",
        file_content="x = 1",
        diff_content="+x = 1",
        config=config,
    )

    assert "ru" in prompt.lower() or "русск" in prompt.lower()


def test_build_prompt_includes_checks():
    config = RepoConfig(checks=[CheckType.SECURITY, CheckType.BUGS])
    prompt = build_review_prompt(
        file_path="main.py",
        file_content="x = 1",
        diff_content="+x = 1",
        config=config,
    )

    assert "security" in prompt.lower()
    assert "bugs" in prompt.lower()
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_prompts.py -v
```

Expected: FAIL (no module)

**Step 3: Create prompt builder**

```python
# src/ai_review/review/prompts.py
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
```

**Step 4: Update review __init__.py**

```python
# src/ai_review/review/__init__.py
from .parser import parse_diff, DiffFile
from .prompts import build_review_prompt

__all__ = ["parse_diff", "DiffFile", "build_review_prompt"]
```

**Step 5: Run test to verify it passes**

```bash
pytest tests/test_prompts.py -v
```

Expected: PASS

**Step 6: Commit**

```bash
git add src/ai_review/review/ tests/test_prompts.py
git commit -m "feat: add prompt builder for code review"
```

---

## Task 8: GitLab Platform Client

**Files:**
- Create: `src/ai_review/platforms/__init__.py`
- Create: `src/ai_review/platforms/base.py`
- Create: `src/ai_review/platforms/gitlab.py`
- Create: `tests/test_gitlab_client.py`

**Step 1: Write failing GitLab client test**

```python
# tests/test_gitlab_client.py
import pytest
from ai_review.platforms.gitlab import GitLabClient


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
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_gitlab_client.py -v
```

Expected: FAIL (no module)

**Step 3: Create GitLab client**

```python
# src/ai_review/platforms/__init__.py
from .gitlab import GitLabClient

__all__ = ["GitLabClient"]
```

```python
# src/ai_review/platforms/base.py
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
```

```python
# src/ai_review/platforms/gitlab.py
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
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_gitlab_client.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/ai_review/platforms/ tests/test_gitlab_client.py
git commit -m "feat: add GitLab API client"
```

---

## Task 9: Review Engine (Orchestration)

**Files:**
- Create: `src/ai_review/review/engine.py`
- Modify: `src/ai_review/review/__init__.py`
- Create: `tests/test_engine.py`

**Step 1: Write failing engine test**

```python
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
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_engine.py -v
```

Expected: FAIL (no module)

**Step 3: Create review engine**

```python
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
```

**Step 4: Update review __init__.py**

```python
# src/ai_review/review/__init__.py
from .parser import parse_diff, DiffFile
from .prompts import build_review_prompt
from .engine import ReviewEngine

__all__ = ["parse_diff", "DiffFile", "build_review_prompt", "ReviewEngine"]
```

**Step 5: Add pyyaml to dependencies**

Update `pyproject.toml` dependencies:

```toml
dependencies = [
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "httpx>=0.26.0",
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",
    "unidiff>=0.7.5",
    "pyyaml>=6.0",
]
```

**Step 6: Run test to verify it passes**

```bash
pip install -e ".[dev]"
pytest tests/test_engine.py -v
```

Expected: PASS

**Step 7: Commit**

```bash
git add src/ai_review/review/ pyproject.toml tests/test_engine.py
git commit -m "feat: add review engine for orchestrating code review"
```

---

## Task 10: Webhook Endpoint

**Files:**
- Modify: `src/ai_review/main.py`
- Create: `tests/test_webhook.py`

**Step 1: Write failing webhook test**

```python
# tests/test_webhook.py
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, AsyncMock
from ai_review.main import app


@pytest.mark.asyncio
async def test_webhook_rejects_invalid_token():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/webhook/gitlab",
            headers={"X-Gitlab-Token": "wrong-token"},
            json={"object_kind": "merge_request"}
        )

    assert response.status_code == 401


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
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_webhook.py -v
```

Expected: FAIL

**Step 3: Implement webhook endpoint**

```python
# src/ai_review/main.py
import logging
from contextlib import asynccontextmanager
from functools import lru_cache
from fastapi import FastAPI, Header, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel

from ai_review.config import Settings
from ai_review.models.webhook import GitLabMREvent, GitLabNoteEvent
from ai_review.platforms.gitlab import GitLabClient
from ai_review.providers.gemini import GeminiProvider
from ai_review.providers.yandex import YandexProvider
from ai_review.review.engine import ReviewEngine


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@lru_cache
def get_settings() -> Settings:
    return Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AI Review Manager starting...")
    yield
    logger.info("AI Review Manager shutting down...")


app = FastAPI(title="AI Review Manager", lifespan=lifespan)


class WebhookResponse(BaseModel):
    status: str
    message: str | None = None


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


@app.post("/webhook/gitlab", response_model=WebhookResponse)
async def gitlab_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_gitlab_token: str = Header(...),
):
    settings = get_settings()

    # Verify webhook token
    if x_gitlab_token != settings.gitlab_webhook_secret:
        raise HTTPException(status_code=401, detail="Invalid webhook token")

    body = await request.json()
    object_kind = body.get("object_kind")

    if object_kind == "merge_request":
        event = GitLabMREvent(**body)
        action = event.object_attributes.action

        if action in ("open", "update"):
            background_tasks.add_task(
                run_review,
                project_id=event.project.id,
                mr_iid=event.object_attributes.iid,
                source_branch=event.object_attributes.source_branch,
            )
            return WebhookResponse(status="accepted", message="Review scheduled")

    elif object_kind == "note":
        event = GitLabNoteEvent(**body)

        if (
            event.object_attributes.noteable_type == "MergeRequest"
            and event.merge_request
            and "/review" in event.object_attributes.note
        ):
            background_tasks.add_task(
                run_review,
                project_id=event.project.id,
                mr_iid=event.merge_request.iid,
                source_branch=event.merge_request.source_branch,
            )
            return WebhookResponse(status="accepted", message="Review scheduled")

    return WebhookResponse(status="ignored", message="Event not relevant")


async def run_review(project_id: int, mr_iid: int, source_branch: str):
    """Background task to run the review."""
    settings = get_settings()

    gitlab = GitLabClient(token=settings.gitlab_token)

    # Select provider
    if settings.default_provider == "gemini" and settings.gemini_api_key:
        provider = GeminiProvider(api_key=settings.gemini_api_key)
    elif settings.default_provider == "yandex" and settings.yandex_api_key:
        provider = YandexProvider(
            api_key=settings.yandex_api_key,
            folder_id=settings.yandex_folder_id or "",
        )
    else:
        logger.error("No LLM provider configured")
        return

    engine = ReviewEngine(gitlab=gitlab, provider=provider)

    try:
        await engine.review_mr(
            project_id=project_id,
            mr_iid=mr_iid,
            source_branch=source_branch,
        )
        logger.info(f"Review completed for MR !{mr_iid}")
    except Exception as e:
        logger.exception(f"Review failed for MR !{mr_iid}: {e}")
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_webhook.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/ai_review/main.py tests/test_webhook.py
git commit -m "feat: add GitLab webhook endpoint with background review"
```

---

## Task 11: Dockerfile

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `.env.example`

**Step 1: Create Dockerfile**

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy source
COPY src/ src/

# Run
ENV PYTHONPATH=/app/src
EXPOSE 8000

CMD ["uvicorn", "ai_review.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Step 2: Create docker-compose.yml**

```yaml
# docker-compose.yml
version: "3.8"

services:
  ai-review:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    restart: unless-stopped
```

**Step 3: Create .env.example**

```bash
# .env.example

# Required: GitLab
GITLAB_TOKEN=your-gitlab-token
GITLAB_WEBHOOK_SECRET=your-webhook-secret

# LLM Providers (at least one required)
GEMINI_API_KEY=your-gemini-key
YANDEX_API_KEY=your-yandex-key
YANDEX_FOLDER_ID=your-folder-id

# Optional
DEFAULT_PROVIDER=gemini
DEFAULT_LANGUAGE=en
LOG_LEVEL=INFO
```

**Step 4: Test Docker build**

```bash
docker build -t ai-review-manager .
```

Expected: Build succeeds

**Step 5: Commit**

```bash
git add Dockerfile docker-compose.yml .env.example
git commit -m "feat: add Docker configuration"
```

---

## Task 12: Run All Tests

**Step 1: Run full test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: All tests pass

**Step 2: Create .gitignore**

```
# .gitignore
__pycache__/
*.py[cod]
.env
.venv/
venv/
*.egg-info/
dist/
build/
.pytest_cache/
.coverage
htmlcov/
```

**Step 3: Final commit**

```bash
git add .gitignore
git commit -m "chore: add gitignore"
git push origin main
```

---

## Summary

After completing all tasks you will have:

1. **FastAPI service** with `/health` and `/webhook/gitlab` endpoints
2. **LLM providers** for Gemini and Yandex GPT
3. **GitLab integration** for fetching diffs and posting comments
4. **Review engine** that orchestrates the full review flow
5. **Docker setup** for easy deployment
6. **Test coverage** for all major components

**Next steps after MVP:**
- Add GitHub support
- Add Jira/Confluence integration
- Add Redis queue for async processing
- Add retry logic and better error handling
