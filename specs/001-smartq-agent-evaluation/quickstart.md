# Quickstart: SmartQ Agent Evaluation

## Prerequisites

1. Install the locked project environment:

   ```bash
   uv sync
   ```

2. Copy `.env.example` to `.env` and set the SmartQ values required by the [CLI contract](contracts/evaluation-cli.md). Use the backend service URL, not the documentation or UI URL.

3. Confirm a JSON-array or JSONL validation set is available:

   ```bash
   test -f dataset/smartq_qa_1.json
   # or
   test -f dataset/smartq_qa_50.jsonl
   ```

## Run a complete evaluation

Use the default Agent QA mode (`POST /agent-chat/:session_id`):

```bash
uv run python -m src.evals.cli run \
  --dataset dataset/smartq_qa_1.json \
  --run-id smartq-qa-1-20260819 \
  --output-dir evaluation_runs \
  --metrics all
```

Use knowledge-base QA mode (`POST /knowledge-chat/:session_id`):

```bash
uv run python -m src.evals.cli run \
  --dataset dataset/smartq_qa_1.json \
  --run-id smartq-knowledge-qa-20260820 \
  --qa-mode knowledge \
  --output-dir evaluation_runs \
  --metrics all
```

Set `SMARTQ_AGENT_ID` to the knowledge-chat profile (for example `builtin-quick-answer`) and `SMARTQ_MODEL_ID` to the summary model (for example `builtin-llm-qwen3-32b`). Set `SMARTQ_KNOWLEDGE_BASE_IDS` and optionally `SMARTQ_KNOWLEDGE_IDS` as comma-separated IDs. Knowledge mode sends `agent_enabled=false`, `web_search_enabled=false`, and `channel=api`.


JSONL input is also accepted directly:

```bash
uv run python -m src.evals.cli run \
  --dataset dataset/smartq_qa_50.jsonl \
  --run-id smartq-jsonl-evaluation \
  --qa-mode knowledge \
  --metrics all
```

Each non-empty JSONL line must contain one complete validation-record JSON object.

Expected artifacts:

```text
evaluation_runs/smartq-qa-1-20260819/
├── run.json
├── records.jsonl
├── metrics.json
└── report.md
```

`records.jsonl` has one terminal record for every source question. A failed Agent request remains represented by a `failed` record; a successful answer without reference indexes is marked retrieval-unscorable but can still receive generation scores.

## Re-score a saved run without SmartQ access

```bash
uv run python -m src.evals.cli score \
  --run-dir evaluation_runs/smartq-qa-1-20260819 \
  --metrics precision@1,recall@5,answer_character_f1
uv run python -m src.evals.cli report \
  --run-dir evaluation_runs/smartq-qa-1-20260819
```

The timestamps and agent calls remain unchanged; only `metrics.json` and `report.md` are regenerated.

Retrieval metrics use an inclusive chunk-index tolerance of ±5. For example, expected index `50` matches retrieved indexes `45` through `55`. Each expected chunk can be matched only once. The tolerance is written to `metrics.json` and `report.md`.


## Validate implementation locally

```bash
uv run pytest
```

Tests use mocked Agent QA and knowledge-chat SSE responses and fixture validation records. They do not contact SmartQ, load local LLMs, or require a GPU.
