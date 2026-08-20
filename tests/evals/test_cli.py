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
            "--ignore-chunk-index",
        ]
    ) == 0

    run_dir = tmp_path / "runs" / "run-1"
    assert (run_dir / "records.jsonl").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "report.md").is_file()
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["chunk_index_comparison_ignored"] is True
    assert metrics["selected_metrics"] == ["answer_exact_match"]


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


def test_knowledge_mode_reuses_agent_id_and_model_id_settings(monkeypatch):
    captured = {}

    class FakeKnowledgeClient:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

    monkeypatch.setattr(cli, "SmartQKnowledgeQAClient", FakeKnowledgeClient)
    monkeypatch.setattr(cli.settings, "SMARTQ_API_URL", "http://smartq.example")
    monkeypatch.setattr(cli.settings, "SMARTQ_API_KEY", "secret-key")
    monkeypatch.setattr(cli.settings, "SMARTQ_TENANT_ID", "tenant-1")
    monkeypatch.setattr(cli.settings, "SMARTQ_AGENT_ID", "builtin-quick-answer")
    monkeypatch.setattr(cli.settings, "SMARTQ_MODEL_ID", "builtin-llm-qwen3-32b")
    monkeypatch.setattr(cli.settings, "SMARTQ_KNOWLEDGE_BASE_IDS", "kb-1")
    monkeypatch.setattr(cli.settings, "SMARTQ_KNOWLEDGE_IDS", "knowledge-1")

    client = cli._qa_client("knowledge")

    assert isinstance(client, FakeKnowledgeClient)
    assert captured["kwargs"]["agent_id"] == "builtin-quick-answer"
    assert captured["kwargs"]["summary_model_id"] == "builtin-llm-qwen3-32b"
    assert captured["kwargs"]["knowledge_base_ids"] == ["kb-1"]
    assert captured["kwargs"]["knowledge_ids"] == ["knowledge-1"]


def test_run_command_accepts_jsonl_validation_dataset(tmp_path, monkeypatch):
    dataset = tmp_path / "validation.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "type": "factual",
                "language": "cn",
                "article_title": "knowledge.doc",
                "chunks": [51],
                "question": "question",
                "answer": "4041人",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "_qa_client", lambda qa_mode: SuccessfulFakeAgent())

    assert cli.main(
        [
            "run",
            "--dataset",
            str(dataset),
            "--run-id",
            "jsonl-run",
            "--output-dir",
            str(tmp_path / "runs"),
            "--metrics",
            "precision@1",
        ]
    ) == 0
    assert (tmp_path / "runs" / "jsonl-run" / "records.jsonl").is_file()


def test_score_command_can_ignore_chunk_index_metrics(tmp_path):
    run_dir = tmp_path / "run-ignore"
    run_dir.mkdir()
    record = AgentResultRecord(
        run_id="run-ignore",
        record_index=0,
        type="factual",
        language="cn",
        article_title="knowledge.doc",
        question="question",
        reference_answer="4041人",
        expected_chunk_indices=[51],
        answer="4041人",
        retrieved_chunk_indices=[999],
        status="success",
    )
    (run_dir / "records.jsonl").write_text(
        record.model_dump_json() + "\n", encoding="utf-8"
    )

    assert cli.main(
        [
            "score",
            "--run-dir",
            str(run_dir),
            "--metrics",
            "all",
            "--ignore-chunk-index",
        ]
    ) == 0

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["chunk_index_comparison_ignored"] is True
    assert metrics["selected_metrics"] == [
        "answer_exact_match",
        "answer_character_f1",
    ]
