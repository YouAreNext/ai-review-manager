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
