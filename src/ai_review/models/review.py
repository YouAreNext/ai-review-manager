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
