"""Collection, persistence, scoring, and reporting for SmartQ QA runs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from hashlib import sha256
from datetime import datetime, timezone
import json
from pathlib import Path
from time import monotonic
from typing import Protocol

from ..lib.smartq import SmartQChunkMapping, extract_chunk_mappings

from .metrics import (
    CHUNK_INDEX_TOLERANCE,
    RETRIEVAL_METRICS,
    answer_character_f1,
    answer_exact_match,
    select_metrics,
)
from .models import (
    AgentEvent,
    AgentResultRecord,
    EvaluationError,
    EvaluationRun,
    EvaluationScore,
    MetricSummary,
    RetrievedChunkMapping,
    ValidationRecord,
    run_directory,
)


RUN_FILENAME = "run.json"
RECORDS_FILENAME = "records.jsonl"
METRICS_FILENAME = "metrics.json"
REPORT_FILENAME = "report.md"


@dataclass(frozen=True)
class AgentResponse:
    """Transport-neutral terminal SmartQ QA response."""

    answer: str
    retrieved_chunk_indices: list[int] | None
    events: list[AgentEvent]
    duration_ms: int
    retrieved_chunks: list[SmartQChunkMapping] = field(default_factory=list)
    error: str | None = None


class AgentClient(Protocol):
    """Minimal boundary used by collection and fully replaceable in tests."""

    def ask(self, question: str) -> AgentResponse:
        """Send one question and return its terminal QA response."""


def load_validation_records(path: Path) -> list[ValidationRecord]:
    """Validate and load a JSON-array Q/A validation set before any network call."""
    try:
        raw_data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise EvaluationError(f"validation dataset does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise EvaluationError(f"validation dataset is not valid JSON: {error}") from error
    if not isinstance(raw_data, list) or not raw_data:
        raise EvaluationError("validation dataset must be a non-empty JSON array")

    records: list[ValidationRecord] = []
    for index, raw_record in enumerate(raw_data):
        if not isinstance(raw_record, dict):
            raise EvaluationError(f"validation record {index} must be a JSON object")
        try:
            records.append(ValidationRecord(record_index=index, **raw_record))
        except ValueError as error:
            raise EvaluationError(f"validation record {index} is invalid: {error}") from error
    return records


def create_run_directory(output_dir: Path, run_id: str) -> Path:
    """Create a new run directory and refuse accidental overwrite."""
    directory = run_directory(output_dir, run_id)
    if directory.exists():
        raise EvaluationError(
            f"evaluation run directory already exists: {directory}; choose a new run ID"
        )
    directory.mkdir(parents=True)
    return directory


def _normalise_article_title(title: str) -> str:
    """Normalize .doc/.docx filename differences for validation mapping."""
    name = Path(title.strip()).name.casefold()
    for suffix in (".docx", ".doc"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _convert_chunk_mappings(
    mappings: Iterable[SmartQChunkMapping],
    article_title: str,
    answer: str | None,
) -> list[RetrievedChunkMapping]:
    """Keep mappings compatible with the validation record's source document."""
    expected_title = _normalise_article_title(article_title)
    raw_mappings = list(mappings)
    titled_matches = [
        mapping
        for mapping in raw_mappings
        if mapping.knowledge_title
        and _normalise_article_title(mapping.knowledge_title) == expected_title
    ]
    compatible = titled_matches or [
        mapping for mapping in raw_mappings if not mapping.knowledge_title
    ]
    result: list[RetrievedChunkMapping] = []
    seen: set[str] = set()
    for mapping in compatible:
        if mapping.chunk_id in seen:
            continue
        seen.add(mapping.chunk_id)
        result.append(
            RetrievedChunkMapping(
                **mapping.__dict__,
                cited_in_answer=bool(answer and mapping.chunk_id in answer),
            )
        )
    return result


def _mappings_from_events(
    events: Iterable[AgentEvent], article_title: str, answer: str | None
) -> list[RetrievedChunkMapping]:
    mappings: list[SmartQChunkMapping] = []
    seen: set[str] = set()
    for event in events:
        if event.response_type not in {"references", "tool_result"}:
            continue
        for mapping in extract_chunk_mappings(event.content, event.data):
            if mapping.chunk_id in seen:
                continue
            seen.add(mapping.chunk_id)
            mappings.append(mapping)
    return _convert_chunk_mappings(mappings, article_title, answer)


def _mapping_indices(
    mappings: Iterable[RetrievedChunkMapping],
) -> list[int] | None:
    indices = list(dict.fromkeys(mapping.chunk_index for mapping in mappings))
    return indices or None


def _result_from_record(
    record: ValidationRecord,
    run_id: str,
    status: str,
    duration_ms: int,
    answer: str | None = None,
    retrieved_chunk_indices: list[int] | None = None,
    retrieved_chunks: list[RetrievedChunkMapping] | None = None,
    events: list[AgentEvent] | None = None,
    error: str | None = None,
) -> AgentResultRecord:
    return AgentResultRecord(
        run_id=run_id,
        record_index=record.record_index,
        type=record.type,
        language=record.language,
        article_title=record.article_title,
        question=record.question,
        reference_answer=record.reference_answer,
        expected_chunk_indices=record.expected_chunk_indices,
        status=status,
        duration_ms=max(duration_ms, 0),
        answer=answer,
        retrieved_chunk_indices=retrieved_chunk_indices,
        retrieved_chunks=retrieved_chunks or [],
        events=events or [],
        error=error,
    )


def collect_records(
    records: Iterable[ValidationRecord],
    run_id: str,
    records_path: Path,
    agent_client: AgentClient,
) -> list[AgentResultRecord]:
    """Collect records sequentially and append one terminal JSON line per input."""
    result_records: list[AgentResultRecord] = []
    with records_path.open("x", encoding="utf-8") as artifact:
        for record in records:
            started_at = monotonic()
            try:
                response = agent_client.ask(record.question)
                duration_ms = response.duration_ms or int(
                    (monotonic() - started_at) * 1000
                )
                retrieved_chunks = _convert_chunk_mappings(
                    getattr(response, "retrieved_chunks", []),
                    record.article_title,
                    response.answer,
                )
                retrieved_chunk_indices = (
                    _mapping_indices(retrieved_chunks)
                    or response.retrieved_chunk_indices
                )
                if response.error:
                    result = _result_from_record(
                        record,
                        run_id,
                        "failed",
                        duration_ms,
                        answer=response.answer or None,
                        retrieved_chunk_indices=retrieved_chunk_indices,
                        retrieved_chunks=retrieved_chunks,
                        events=response.events,
                        error=response.error,
                    )
                elif not response.answer.strip():
                    result = _result_from_record(
                        record,
                        run_id,
                        "invalid_response",
                        duration_ms,
                        retrieved_chunk_indices=retrieved_chunk_indices,
                        retrieved_chunks=retrieved_chunks,
                        events=response.events,
                        error="SmartQ QA response did not contain an answer",
                    )
                else:
                    result = _result_from_record(
                        record,
                        run_id,
                        "success",
                        duration_ms,
                        answer=response.answer,
                        retrieved_chunk_indices=retrieved_chunk_indices,
                        retrieved_chunks=retrieved_chunks,
                        events=response.events,
                    )
            except Exception as error:  # per-record failures must not abort collection
                result = _result_from_record(
                    record,
                    run_id,
                    "failed",
                    int((monotonic() - started_at) * 1000),
                    error=f"{type(error).__name__}: {error}",
                )
            artifact.write(result.model_dump_json() + "\n")
            artifact.flush()
            result_records.append(result)
    return result_records


def _write_run_manifest(path: Path, run: EvaluationRun) -> None:
    path.write_text(run.model_dump_json(indent=2) + "\n", encoding="utf-8")


def collect_dataset(
    dataset_path: Path,
    output_dir: Path,
    run_id: str,
    agent_client: AgentClient,
) -> tuple[Path, list[AgentResultRecord]]:
    """Load a dataset, persist run metadata, and collect records sequentially."""
    records = load_validation_records(dataset_path)
    directory = create_run_directory(output_dir, run_id)
    run = EvaluationRun(
        run_id=run_id,
        dataset_path=str(dataset_path),
        dataset_sha256=dataset_sha256(dataset_path),
        input_count=len(records),
        agent_id=str(getattr(agent_client, "agent_id", "configured-agent")),
        qa_mode=str(getattr(agent_client, "qa_mode", "agent")),
        knowledge_base_ids=list(getattr(agent_client, "knowledge_base_ids", [])),
        started_at=datetime.now(timezone.utc).isoformat(),
        status="running",
    )
    manifest_path = directory / RUN_FILENAME
    _write_run_manifest(manifest_path, run)
    results = collect_records(
        records,
        run_id,
        directory / RECORDS_FILENAME,
        agent_client,
    )
    terminal_status = (
        "completed_with_failures"
        if any(result.status != "success" for result in results)
        else "completed"
    )
    _write_run_manifest(
        manifest_path,
        run.model_copy(
            update={
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "status": terminal_status,
            }
        ),
    )
    return directory, results


def load_result_records(path: Path) -> list[AgentResultRecord]:
    """Load a non-empty JSONL artifact without contacting SmartQ."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise EvaluationError(f"saved result artifact does not exist: {path}") from error
    if not lines:
        raise EvaluationError(f"saved result artifact is empty: {path}")

    records: list[AgentResultRecord] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            record = AgentResultRecord.model_validate_json(line)
            if not record.retrieved_chunks:
                mappings = _mappings_from_events(
                    record.events, record.article_title, record.answer
                )
                if mappings:
                    record = record.model_copy(
                        update={
                            "retrieved_chunks": mappings,
                            "retrieved_chunk_indices": _mapping_indices(mappings),
                        }
                    )
            records.append(record)
        except ValueError as error:
            raise EvaluationError(
                f"saved result artifact has invalid JSONL at line {line_number}: {error}"
            ) from error
    return records


def score_records(
    records: list[AgentResultRecord], raw_metrics: str | Iterable[str] = "all"
) -> EvaluationScore:
    """Calculate selected deterministic metrics from saved terminal records."""
    if not records:
        raise EvaluationError("cannot score an empty result set")
    selected_metrics = select_metrics(raw_metrics)
    run_ids = {record.run_id for record in records}
    if len(run_ids) != 1:
        raise EvaluationError("saved result records must belong to exactly one run")

    retrieval_records = [
        record
        for record in records
        if record.status == "success" and record.retrieved_chunk_indices is not None
    ]
    generation_records = [
        record
        for record in records
        if record.status == "success" and record.answer is not None
    ]
    summaries: list[MetricSummary] = []

    for metric_name in selected_metrics:
        if metric_name == "answer_exact_match":
            values = [
                answer_exact_match(record.answer or "", record.reference_answer)
                for record in generation_records
            ]
            summaries.append(
                MetricSummary(
                    metric_name=metric_name,
                    value=sum(values) / len(values) if values else None,
                    scored_count=len(values),
                    unscorable_count=len(records) - len(values),
                )
            )
            continue
        if metric_name == "answer_character_f1":
            values = [
                answer_character_f1(record.answer or "", record.reference_answer)
                for record in generation_records
            ]
            summaries.append(
                MetricSummary(
                    metric_name=metric_name,
                    value=sum(values) / len(values) if values else None,
                    scored_count=len(values),
                    unscorable_count=len(records) - len(values),
                )
            )
            continue

        operation, raw_cutoff = metric_name.split("@", maxsplit=1)
        cutoff = int(raw_cutoff)
        if retrieval_records:
            predictions = [
                record.retrieved_chunk_indices or []
                for record in retrieval_records
            ]
            expected = [record.expected_chunk_indices for record in retrieval_records]
            value = float(RETRIEVAL_METRICS[operation](predictions, expected, [cutoff])[0])
        else:
            value = None
        summaries.append(
            MetricSummary(
                metric_name=metric_name,
                value=value,
                scored_count=len(retrieval_records),
                unscorable_count=len(records) - len(retrieval_records),
            )
        )

    return EvaluationScore.now(
        run_id=next(iter(run_ids)),
        selected_metrics=selected_metrics,
        input_count=len(records),
        success_count=sum(record.status == "success" for record in records),
        failed_count=sum(record.status == "failed" for record in records),
        invalid_response_count=sum(
            record.status == "invalid_response" for record in records
        ),
        retrieval_index_tolerance=CHUNK_INDEX_TOLERANCE,
        metrics=summaries,
    )


def score_run(run_dir: Path, raw_metrics: str | Iterable[str] = "all") -> EvaluationScore:
    """Score `records.jsonl` and write a new machine-readable metrics artifact."""
    records = load_result_records(run_dir / RECORDS_FILENAME)
    score = score_records(records, raw_metrics)
    (run_dir / METRICS_FILENAME).write_text(
        score.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return score


def load_score(path: Path) -> EvaluationScore:
    """Load a previously written score artifact."""
    try:
        return EvaluationScore.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise EvaluationError(f"metrics artifact does not exist: {path}") from error
    except ValueError as error:
        raise EvaluationError(f"metrics artifact is invalid: {error}") from error


def render_report(
    score: EvaluationScore,
    records: list[AgentResultRecord],
    run: EvaluationRun | None = None,
) -> str:
    """Render a traceable Markdown report from saved results and aggregates."""
    lines = [
        f"# SmartQ QA Evaluation Report: {score.run_id}",
        "",
        "## Run Summary",
        "",
    ]
    if run:
        lines.extend(
            [
                f"- Dataset: {run.dataset_path}",
                f"- Dataset SHA-256: {run.dataset_sha256}",
                f"- QA mode: {run.qa_mode}",
                f"- Agent: {run.agent_id}",
                f"- Run status: {run.status}",
            ]
        )
    lines.extend(
        [
            f"- Report generated: {score.generated_at}",
            f"- Selected metrics: {', '.join(score.selected_metrics)}",
            f"- Retrieval chunk-index tolerance: ±{score.retrieval_index_tolerance}",
            f"- Input records: {score.input_count}",
            f"- Successful records: {score.success_count}",
            f"- Failed records: {score.failed_count}",
            f"- Invalid responses: {score.invalid_response_count}",
            "",
            "## Metrics",
            "",
            "| Metric | Value | Scored | Unscorable |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric in score.metrics:
        value = "N/A" if metric.value is None else f"{metric.value:.6f}"
        lines.append(
            f"| {metric.metric_name} | {value} | {metric.scored_count} | "
            f"{metric.unscorable_count} |"
        )

    lines.extend(["", "## Per-question diagnostics", ""])
    for record in records:
        diagnostic = record.error or ""
        if record.status == "success" and record.retrieved_chunk_indices is None:
            diagnostic = "Retrieval evidence unavailable"
        relevant_mappings = [
            mapping
            for mapping in record.retrieved_chunks
            if mapping.chunk_index in set(record.expected_chunk_indices)
        ]
        mapping_summary = ", ".join(
            f"{mapping.chunk_index} -> {mapping.chunk_id}"
            + (" (cited)" if mapping.cited_in_answer else "")
            for mapping in relevant_mappings
        )
        lines.extend(
            [
                f"### [{record.record_index}] {record.question}",
                "",
                f"- Status: {record.status}",
                f"- Retrieval eligible: {record.retrieved_chunk_indices is not None}",
                f"- Generation eligible: {record.status == 'success' and bool(record.answer)}",
                f"- Expected chunk indices: {record.expected_chunk_indices}",
                f"- Ground-truth chunk mappings: {mapping_summary or 'Not observed'}",
                f"- Diagnostic: {diagnostic or 'None'}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"

def report_run(run_dir: Path) -> Path:
    """Render the human-readable report using only saved local artifacts."""
    records = load_result_records(run_dir / RECORDS_FILENAME)
    score = load_score(run_dir / METRICS_FILENAME)
    manifest_path = run_dir / RUN_FILENAME
    run = None
    if manifest_path.exists():
        try:
            run = EvaluationRun.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
        except ValueError as error:
            raise EvaluationError(f"run manifest is invalid: {error}") from error
    report_path = run_dir / REPORT_FILENAME
    report_path.write_text(render_report(score, records, run), encoding="utf-8")
    return report_path


def dataset_sha256(path: Path) -> str:
    """Return a reproducibility hash for user-visible run diagnostics."""
    return sha256(path.read_bytes()).hexdigest()
