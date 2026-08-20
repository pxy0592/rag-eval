# Data Model: SmartQ Agent Evaluation

## ValidationRecord

Represents one source item from the input JSON array.

| Field | Type | Rules |
|---|---|---|
| `record_index` | integer | Source-array position; stable identity within a run. |
| `type` | string | Required non-empty Q/A type. |
| `language` | string | Required non-empty language. |
| `article_title` | string | Required non-empty source title. |
| `expected_chunk_indices` | list of integers | Required; each value is non-negative; preserves source order. |
| `question` | string | Required non-empty question. |
| `reference_answer` | string | Required non-empty answer. |

## EvaluationRun

Represents one local invocation and its immutable source selection.

| Field | Type | Rules |
|---|---|---|
| `run_id` | string | User-supplied or generated; safe filesystem segment. |
| `dataset_path` | string | Local source path recorded without secrets. |
| `dataset_sha256` | string | Content hash for reproducibility. |
| `selected_metrics` | list of strings | Validated against the metric registry. |
| `agent_id` | string | Configured target; no API key. |
| `knowledge_base_ids` | list of strings | Optional configured scope. |
| `started_at` / `completed_at` | timestamp | Run lifecycle timestamps. |
| `status` | enum | `running`, `completed`, or `completed_with_failures`. |

## AgentResultRecord

Append-only persisted output for one `ValidationRecord`.

| Field | Type | Rules |
|---|---|---|
| `run_id`, `record_index` | identifiers | Link result to exactly one run and source record. |
| `question`, `reference_answer`, `expected_chunk_indices` | copied source evidence | Required for independent later scoring. |
| `answer` | string or null | Concatenated `answer` SSE payload when available. |
| `retrieved_chunk_indices` | list of integers or null | Ranked, valid, de-duplicated `references` indexes; null means evidence absent. |
| `status` | enum | `success`, `failed`, or `invalid_response`. |
| `duration_ms` | integer | Non-negative elapsed duration. |
| `error` | string or null | Sanitized actionable error; never credentials. |
| `events` | list of sanitized event summaries | Records response type and only non-sensitive diagnostic content needed for audit. |

Transitions: `pending` → `success` / `failed` / `invalid_response`; persisted terminal records are immutable. A run is complete only after it has a terminal result for every source record.

## MetricSummary

| Field | Type | Rules |
|---|---|---|
| `metric_name` | string | Registered metric identifier. |
| `value` | number or null | Aggregate score, null only when no record is eligible. |
| `scored_count` | integer | Number of records used as the denominator. |
| `unscorable_count` | integer | Records excluded due to absent required evidence. |

## EvaluationReport

Derived artifact containing run metadata, input/terminal record counts, selected metrics, retrieval summaries, generation summaries, and per-record status/eligibility diagnostics. It is written as `metrics.json` and `report.md` and can be regenerated from `records.jsonl`.
