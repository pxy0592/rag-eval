"""Command-line entry point for deterministic SmartQ QA evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

from ..lib.smartq import (
    SmartQAPIError,
    SmartQAgentClient,
    SmartQKnowledgeQAClient,
)
from ..settings import settings
from .models import EvaluationError
from .runner import collect_dataset, report_run, score_run


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate SmartQ Agent or knowledge-base QA performance on a dataset.\n"
            "Execute the full evaluation workflow (collect, score, report) or individual steps.\n"
            "- Collect question-answer pairs from a dataset and send them through the selected SmartQ QA mode.\n"
            "- Score the evaluation results and generate a metrics report.\n"
            "- Review the evaluation results in the generated report."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def collection_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("--dataset", type=Path, required=True)
        command.add_argument("--run-id")
        command.add_argument("--output-dir", type=Path, default=Path("evaluation_runs"))
        command.add_argument(
            "--qa-mode",
            choices=("agent", "knowledge"),
            default="agent",
            help=(
                "Submit questions through /agent-chat (default) or "
                "/knowledge-chat."
            ),
        )
        command.add_argument("--metrics", default="all")

    collection_arguments(subparsers.add_parser("collect"))

    score = subparsers.add_parser("score")
    score.add_argument("--run-dir", type=Path, required=True)
    score.add_argument("--metrics", default="all")

    report = subparsers.add_parser("report")
    report.add_argument("--run-dir", type=Path, required=True)

    collection_arguments(subparsers.add_parser("run"))
    return parser


def _default_run_id(qa_mode: str) -> str:
    return f"smartq-{qa_mode}-" + datetime.now(timezone.utc).strftime(
        "%Y%m%d-%H%M%S"
    )


def _configured_ids(raw_value: str) -> list[str]:
    return [value.strip() for value in raw_value.split(",") if value.strip()]


def _qa_client(
    qa_mode: str,
) -> SmartQAgentClient | SmartQKnowledgeQAClient:
    knowledge_base_ids = _configured_ids(settings.SMARTQ_KNOWLEDGE_BASE_IDS)
    if qa_mode == "knowledge":
        return SmartQKnowledgeQAClient(
            settings.SMARTQ_API_URL,
            settings.SMARTQ_API_KEY,
            settings.SMARTQ_TENANT_ID,
            knowledge_base_ids=knowledge_base_ids,
            knowledge_ids=_configured_ids(settings.SMARTQ_KNOWLEDGE_IDS),
            timeout_seconds=settings.SMARTQ_AGENT_TIMEOUT_SECONDS,
        )
    return SmartQAgentClient(
        settings.SMARTQ_API_URL,
        settings.SMARTQ_API_KEY,
        settings.SMARTQ_TENANT_ID,
        settings.SMARTQ_AGENT_ID,
        knowledge_base_ids=knowledge_base_ids,
        timeout_seconds=settings.SMARTQ_AGENT_TIMEOUT_SECONDS,
    )


def main(argv: list[str] | None = None) -> int:
    """Run collection, scoring, reporting, or the complete evaluation workflow."""
    parser = _parser()
    try:
        arguments = parser.parse_args(argv)
    except SystemExit as error:
        return int(error.code)

    try:
        if arguments.command == "collect":
            run_id = arguments.run_id or _default_run_id(arguments.qa_mode)
            run_dir, records = collect_dataset(
                arguments.dataset,
                arguments.output_dir,
                run_id,
                _qa_client(arguments.qa_mode),
            )
            print(f"Collected {len(records)} records: {run_dir}")
            return 0
        if arguments.command == "score":
            score = score_run(arguments.run_dir, arguments.metrics)
            print(arguments.run_dir / "metrics.json")
            print(f"Scored {score.input_count} records")
            return 0
        if arguments.command == "report":
            print(report_run(arguments.run_dir))
            return 0
        if arguments.command == "run":
            run_id = arguments.run_id or _default_run_id(arguments.qa_mode)
            run_dir, records = collect_dataset(
                arguments.dataset,
                arguments.output_dir,
                run_id,
                _qa_client(arguments.qa_mode),
            )
            score_run(run_dir, arguments.metrics)
            report_path = report_run(run_dir)
            print(f"Collected {len(records)} records: {run_dir}")
            print(report_path)
            return 0
        raise EvaluationError(f"unsupported command: {arguments.command}")
    except (EvaluationError, SmartQAPIError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
