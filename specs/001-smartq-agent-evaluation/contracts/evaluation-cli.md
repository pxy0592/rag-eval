# Evaluation CLI Contract

Run commands through the package:

```bash
uv run python -m src.evals.cli <command> [options]
```

## Shared configuration

The CLI reads only environment configuration; options and artifacts must never print secret values.

| Variable | Required | Meaning |
|---|---|---|
| `SMARTQ_API_URL` | yes | SmartQ backend base URL, with or without `/api/v1`. |
| `SMARTQ_API_KEY` | yes | SmartQ API key sent as `X-API-Key`. |
| `SMARTQ_TENANT_ID` | yes | SmartQ tenant sent as `X-Tenant-ID`. |
| `SMARTQ_AGENT_ID` | yes | Agent used in the Agent QA request. |
| `SMARTQ_KNOWLEDGE_BASE_IDS` | no | Comma-separated KB IDs sent in `knowledge_base_ids`. |
| `SMARTQ_AGENT_TIMEOUT_SECONDS` | no | Per-question timeout; defaults to 180 seconds. |

## Commands

### `collect`

```bash
uv run python -m src.evals.cli collect \
  --dataset dataset/smartq_qa_1.json \
  --run-id smoke-20260819 \
  --output-dir evaluation_runs
```

Validates input before the first request, then creates `evaluation_runs/<run-id>/records.jsonl`. It makes one session and one Agent QA request for each source record in order. Exit non-zero only when setup/input prevents a run; individual request failures are recorded and summarized.

### `score`

```bash
uv run python -m src.evals.cli score \
  --run-dir evaluation_runs/smoke-20260819 \
  --metrics all
```

Reads only saved `records.jsonl`, validates a comma-separated metric subset or `all`, and writes `metrics.json`. Supported metrics are the registry's retrieval metrics (`precision@k`, `recall@k`, `mrr@k`, `ndcg@k`, `map@k`) and generation metrics (`answer_exact_match`, `answer_character_f1`).

### `report`

```bash
uv run python -m src.evals.cli report \
  --run-dir evaluation_runs/smoke-20260819
```

Reads saved records and `metrics.json`, writes `report.md`, and prints its path. It never calls SmartQ.

### `run`

```bash
uv run python -m src.evals.cli run \
  --dataset dataset/smartq_qa_1.json \
  --run-id smoke-20260819 \
  --metrics precision@1,recall@5,answer_character_f1 \
  --output-dir evaluation_runs
```

Performs `collect`, `score`, and `report` in order. If `--metrics` is omitted or `all`, it selects all registered metrics.

## Output and exit behavior

- `run.json` records the dataset identity, content hash, safe Agent scope, and terminal run status. JSONL preserves source order and contains one terminal record per source input record.
- `metrics.json` includes the metric value and scored/failed/unscorable counts; no eligible denominator produces a null value rather than a fabricated zero.
- `report.md` identifies dataset, run ID, selected metrics, aggregate results, coverage, and each non-success or unscorable record.
- Invalid data, missing required configuration, corrupt saved artifacts, invalid run ID, and unsupported metrics produce a clear non-zero CLI error before silent partial output.
