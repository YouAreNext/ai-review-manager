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
