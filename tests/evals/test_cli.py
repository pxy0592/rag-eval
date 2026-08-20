from src.evals import cli


def test_collect_command_requires_required_arguments(capsys):
    assert cli.main(["collect"]) == 2
    assert "--dataset" in capsys.readouterr().err

from src.evals.models import AgentResultRecord


def test_score_command_writes_metrics_from_saved_records(tmp_path, capsys):
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    record = AgentResultRecord(
        run_id="run-1",
        record_index=0,
        type="factual",
        language="cn",
        article_title="knowledge.doc",
        question="question",
        reference_answer="4041人",
        expected_chunk_indices=[51],
        answer="4041人",
        retrieved_chunk_indices=[51],
        status="success",
    )
    (run_dir / "records.jsonl").write_text(record.model_dump_json() + "\n")

    assert cli.main(["score", "--run-dir", str(run_dir), "--metrics", "precision@1"]) == 0
    assert "metrics.json" in capsys.readouterr().out
    assert (run_dir / "metrics.json").is_file()


def test_score_command_rejects_unknown_metric(tmp_path, capsys):
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    (run_dir / "records.jsonl").write_text("{}\n")

    assert cli.main(["score", "--run-dir", str(run_dir), "--metrics", "bogus"]) == 2
    assert "error:" in capsys.readouterr().err

import json

from src.evals.runner import AgentResponse


class SuccessfulFakeAgent:
    def ask(self, question):
        return AgentResponse(
            answer="4041人",
            retrieved_chunk_indices=[51],
            events=[],
            duration_ms=1,
        )


def test_run_command_collects_scores_and_reports_in_sequence(tmp_path, monkeypatch):
    dataset = tmp_path / "validation.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "type": "factual",
                    "language": "cn",
                    "article_title": "knowledge.doc",
                    "chunks": [51],
                    "question": "question",
                    "answer": "4041人",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "_qa_client", lambda qa_mode: SuccessfulFakeAgent())

    assert cli.main(
        [
            "run",
            "--dataset",
            str(dataset),
            "--run-id",
            "run-1",
            "--output-dir",
            str(tmp_path / "runs"),
            "--metrics",
            "precision@1,answer_exact_match",
        ]
    ) == 0

    run_dir = tmp_path / "runs" / "run-1"
    assert (run_dir / "records.jsonl").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "report.md").is_file()


def test_run_command_selects_knowledge_chat_mode(tmp_path, monkeypatch):
    dataset = tmp_path / "validation.json"
    dataset.write_text(
        json.dumps(
            [{
                "type": "factual",
                "language": "cn",
                "article_title": "knowledge.doc",
                "chunks": [51],
                "question": "question",
                "answer": "4041人",
            }],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    selected_modes = []
    monkeypatch.setattr(
        cli,
        "_qa_client",
        lambda qa_mode: selected_modes.append(qa_mode) or SuccessfulFakeAgent(),
    )

    assert cli.main(
        [
            "run",
            "--dataset",
            str(dataset),
            "--run-id",
            "knowledge-run",
            "--output-dir",
            str(tmp_path / "runs"),
            "--qa-mode",
            "knowledge",
            "--metrics",
            "precision@1",
        ]
    ) == 0
    assert selected_modes == ["knowledge"]
