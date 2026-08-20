"""Data models for deterministic SmartQ Agent evaluation runs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EvaluationError(ValueError):
    """Raised for invalid evaluation inputs, artifacts, or options."""


class ValidationRecord(BaseModel):
    """One Q/A record loaded from a validation-set JSON array."""

    model_config = ConfigDict(populate_by_name=True)

    record_index: int
    type: str
    language: str
    article_title: str
    expected_chunk_indices: list[int] = Field(validation_alias="chunks")
    question: str
    reference_answer: str = Field(validation_alias="answer")

    @field_validator("type", "language", "article_title", "question", "reference_answer")
    @classmethod
    def require_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("expected_chunk_indices")
    @classmethod
    def validate_chunk_indices(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("must contain at least one chunk index")
        if any(isinstance(index, bool) or index < 0 for index in value):
            raise ValueError("must contain non-negative integer chunk indices")
        return value


class AgentEvent(BaseModel):
    """Sanitized SmartQ SSE event retained for local audit diagnostics."""

    response_type: str
    content: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class EvaluationRun(BaseModel):
    """Configuration-safe metadata for one local collection run."""

    run_id: str
    dataset_path: str
    dataset_sha256: str
    input_count: int
    agent_id: str
    knowledge_base_ids: list[str] = Field(default_factory=list)
    started_at: str
    completed_at: str | None = None
    status: Literal["running", "completed", "completed_with_failures"]


class AgentResultRecord(BaseModel):
    """Durable terminal result for one validation record."""

    run_id: str
    record_index: int
    type: str
    language: str
    article_title: str
    question: str
    reference_answer: str
    expected_chunk_indices: list[int]
    answer: str | None = None
    retrieved_chunk_indices: list[int] | None = None
    status: Literal["success", "failed", "invalid_response"]
    duration_ms: int = 0
    error: str | None = None
    events: list[AgentEvent] = Field(default_factory=list)


class MetricSummary(BaseModel):
    """Aggregate score and the denominator used to calculate it."""

    metric_name: str
    value: float | None
    scored_count: int
    unscorable_count: int


class EvaluationScore(BaseModel):
    """Machine-readable aggregate score report for a saved run."""

    run_id: str
    generated_at: str
    selected_metrics: list[str]
    input_count: int
    success_count: int
    failed_count: int
    invalid_response_count: int
    metrics: list[MetricSummary]

    @classmethod
    def now(cls, **kwargs: Any) -> "EvaluationScore":
        return cls(
            generated_at=datetime.now(timezone.utc).isoformat(),
            **kwargs,
        )


_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def validate_run_id(run_id: str) -> str:
    """Return a safe local run ID or raise an actionable input error."""
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise EvaluationError(
            "run ID must contain only letters, numbers, dots, underscores, or hyphens"
        )
    return run_id


def run_directory(output_dir: Path, run_id: str) -> Path:
    """Build the local artifact directory without accepting path traversal."""
    return output_dir / validate_run_id(run_id)
