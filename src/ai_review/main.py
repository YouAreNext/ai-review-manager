# src/ai_review/main.py
import re
import logging
from contextlib import asynccontextmanager
from functools import lru_cache
from fastapi import FastAPI, Header, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel, model_validator

from ai_review.config import Settings
from ai_review.models.webhook import GitLabMREvent, GitLabNoteEvent
from ai_review.platforms.gitlab import GitLabClient
from ai_review.providers.base import LLMProvider
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


class ReviewRequest(BaseModel):
    url: str | None = None
    project_id: int | None = None
    mr_iid: int | None = None

    @model_validator(mode="after")
    def check_params(self):
        if not self.url and not (self.project_id and self.mr_iid):
            raise ValueError("Either url or project_id+mr_iid required")
        return self


class ReviewResponse(BaseModel):
    status: str
    project_id: int | None = None
    mr_iid: int | None = None
    comments_posted: int | None = None
    summary: str | None = None
    error: str | None = None


def parse_gitlab_mr_url(url: str) -> tuple[str, int]:
    """Parse GitLab MR URL -> (project_path, mr_iid)."""
    match = re.match(r"https?://[^/]+/(.+?)/-/merge_requests/(\d+)", url)
    if not match:
        raise ValueError(f"Invalid GitLab MR URL: {url}")
    return match.group(1), int(match.group(2))


def get_provider(settings: Settings) -> LLMProvider | None:
    """Get LLM provider based on settings."""
    if settings.default_provider == "gemini" and settings.gemini_api_key:
        return GeminiProvider(api_key=settings.gemini_api_key)
    elif settings.default_provider == "yandex" and settings.yandex_api_key:
        return YandexProvider(
            api_key=settings.yandex_api_key,
            folder_id=settings.yandex_folder_id or "",
        )
    return None


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


@app.post("/api/review", response_model=ReviewResponse)
async def trigger_review(request: ReviewRequest):
    """Manually trigger a review for a merge request."""
    settings = get_settings()
    gitlab = GitLabClient(token=settings.gitlab_token)

    try:
        # Resolve project_id and mr_iid
        if request.url:
            project_path, mr_iid = parse_gitlab_mr_url(request.url)
            project = await gitlab.get_project_by_path(project_path)
            project_id = project["id"]
        else:
            project_id = request.project_id
            mr_iid = request.mr_iid

        # Get MR info for source_branch
        mr_info = await gitlab.get_mr_info(project_id, mr_iid)
        source_branch = mr_info["source_branch"]

        # Get provider
        provider = get_provider(settings)
        if not provider:
            return ReviewResponse(
                status="error",
                error="No LLM provider configured",
            )

        # Run review
        engine = ReviewEngine(gitlab=gitlab, provider=provider)
        result = await engine.review_mr(
            project_id=project_id,
            mr_iid=mr_iid,
            source_branch=source_branch,
        )

        return ReviewResponse(
            status="completed",
            project_id=project_id,
            mr_iid=mr_iid,
            comments_posted=result.comments_count,
            summary=result.summary or "No issues found",
        )

    except ValueError as e:
        return ReviewResponse(status="error", error=str(e))
    except Exception as e:
        logger.exception(f"Review failed: {e}")
        return ReviewResponse(status="error", error=str(e))


async def run_review(project_id: int, mr_iid: int, source_branch: str):
    """Background task to run the review."""
    settings = get_settings()
    gitlab = GitLabClient(token=settings.gitlab_token)

    provider = get_provider(settings)
    if not provider:
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
