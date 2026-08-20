import json
from pathlib import Path

import pytest

from src.evals.models import EvaluationError
from src.evals.runner import (
    AgentResponse,
    collect_dataset,
    load_result_records,
    load_validation_records,
)


class SequentialFakeAgent:
    def __init__(self):
        self.questions = []
        self.active = 0
        self.max_active = 0

    def ask(self, question):
        self.questions.append(question)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if question == "bad":
                raise RuntimeError("agent unavailable")
            return AgentResponse(
                answer=f"answer:{question}",
                retrieved_chunk_indices=[51],
                events=[],
                duration_ms=12,
            )
        finally:
            self.active -= 1


def write_dataset(path: Path, questions=("first", "bad", "third")):
    path.write_text(
        json.dumps(
            [
                {
                    "type": "factual",
                    "language": "cn",
                    "article_title": "knowledge.doc",
                    "chunks": [51],
                    "question": question,
                    "answer": f"reference:{question}",
                }
                for question in questions
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_collection_is_sequential_and_persists_terminal_record_for_each_input(tmp_path):
    dataset = tmp_path / "validation.json"
    write_dataset(dataset)
    agent = SequentialFakeAgent()

    run_dir, records = collect_dataset(dataset, tmp_path / "runs", "run-1", agent)

    assert agent.questions == ["first", "bad", "third"]
    assert agent.max_active == 1
    assert [record.status for record in records] == ["success", "failed", "success"]
    saved = load_result_records(run_dir / "records.jsonl")
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert [record.record_index for record in saved] == [0, 1, 2]
    assert manifest["status"] == "completed_with_failures"
    assert manifest["dataset_sha256"]
    assert saved[1].error == "RuntimeError: agent unavailable"
    assert len((run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()) == 3


def test_dataset_validation_happens_before_agent_calls(tmp_path):
    dataset = tmp_path / "invalid.json"
    dataset.write_text("[]", encoding="utf-8")

    with pytest.raises(EvaluationError, match="non-empty JSON array"):
        load_validation_records(dataset)


def test_collection_does_not_overwrite_an_existing_run(tmp_path):
    dataset = tmp_path / "validation.json"
    write_dataset(dataset, questions=("first",))
    agent = SequentialFakeAgent()
    collect_dataset(dataset, tmp_path / "runs", "run-1", agent)

    with pytest.raises(EvaluationError, match="already exists"):
        collect_dataset(dataset, tmp_path / "runs", "run-1", agent)

from src.evals.models import AgentResultRecord
from src.evals.runner import score_records, score_run


def saved_record(index, *, status="success", answer="4041人", retrieved=[51]):
    return AgentResultRecord(
        run_id="run-1",
        record_index=index,
        type="factual",
        language="cn",
        article_title="knowledge.doc",
        question=f"question-{index}",
        reference_answer="4041人",
        expected_chunk_indices=[51],
        answer=answer,
        retrieved_chunk_indices=retrieved,
        status=status,
    )


def test_scoring_tracks_retrieval_and_generation_eligibility():
    records = [
        saved_record(0),
        saved_record(1, answer="wrong", retrieved=None),
        saved_record(2, status="failed", answer=None, retrieved=None),
    ]

    score = score_records(
        records,
        "precision@1,recall@1,answer_exact_match,answer_character_f1",
    )
    summaries = {summary.metric_name: summary for summary in score.metrics}

    assert summaries["precision@1"].value == 1.0
    assert summaries["precision@1"].scored_count == 1
    assert summaries["precision@1"].unscorable_count == 2
    assert summaries["answer_exact_match"].value == 0.5
    assert summaries["answer_exact_match"].scored_count == 2
    assert summaries["answer_character_f1"].value == 0.5
    assert score.failed_count == 1


def test_score_run_reads_saved_records_without_an_agent_client(tmp_path):
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    (run_dir / "records.jsonl").write_text(
        saved_record(0).model_dump_json() + "\n", encoding="utf-8"
    )

    score = score_run(run_dir, "precision@1")

    assert score.metrics[0].value == 1.0
    assert (run_dir / "metrics.json").is_file()

from src.evals.runner import report_run


def test_report_includes_metrics_coverage_and_unscorable_diagnostics(tmp_path):
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    records = [
        saved_record(0),
        saved_record(1, answer="wrong", retrieved=None),
        saved_record(2, status="failed", answer=None, retrieved=None),
    ]
    (run_dir / "records.jsonl").write_text(
        "".join(record.model_dump_json() + "\n" for record in records),
        encoding="utf-8",
    )
    score_run(run_dir, "precision@1,answer_exact_match")

    report_path = report_run(run_dir)
    report = report_path.read_text(encoding="utf-8")

    assert report_path.name == "report.md"
    assert "| precision@1 | 1.000000 | 1 | 2 |" in report
    assert "- Selected metrics: precision@1, answer_exact_match" in report
    assert "- Report generated:" in report
    assert "Retrieval evidence unavailable" in report
    assert "Status: failed" in report
    assert "secret" not in report
