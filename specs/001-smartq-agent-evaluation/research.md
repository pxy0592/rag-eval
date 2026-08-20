# Research: SmartQ Agent Evaluation

## Decision: Use SmartQ's session-backed Agent QA SSE API

**Rationale**: Current SmartQ source exposes `POST /api/v1/sessions` to create a session and `POST /api/v1/agent-chat/{session_id}` to run Agent QA. The SSE stream emits `answer`, `references`, `error`, and `complete` events. `references` carry `SearchResult` objects containing `chunk_index`, which matches the integer ground-truth values in `dataset/smartq_qa_1.json`.

**Alternatives considered**:

- Call the lower-level knowledge-search APIs directly: rejected because it does not evaluate the configured SmartQ Agent's complete behavior.
- Score only final answer text: rejected because it cannot measure retrieval quality or distinguish missing retrieval evidence.
- Reuse SmartQ's experimental `dataset/ragas_eval.py`: rejected because it is outside this repository, uses asynchronous/concurrent collection and external LLM metrics, while this feature requires sequential deterministic local scoring.

## Decision: Reuse stdlib urllib and inject the request boundary in tests

**Rationale**: `src/lib/smartq.py` already uses `urllib.request.Request` and `urlopen`, and its tests mock that boundary. Extending it avoids adding an unnecessary direct dependency and allows deterministic SSE fixtures.

**Alternatives considered**:

- Add `httpx`: rejected for v1 because it is not a direct project dependency and does not provide required behavior beyond a small synchronous streaming client.
- Add an SSE client library: rejected because the protocol only needs line-by-line `data:` parsing.

## Decision: Store append-only JSONL per-question records and derived JSON/Markdown reports

**Rationale**: A JSONL record is written immediately after every question, providing durable auditability of partial or failed runs. Re-scoring reads those saved records, so it never calls SmartQ. JSON holds programmatic metrics while Markdown holds an operator-readable summary.

**Alternatives considered**:

- Store only a final aggregate report: rejected because it loses answer/reference/retrieval evidence and cannot be re-scored.
- Store raw API keys or full request headers for reproducibility: rejected because artifacts must not contain credentials.
- Use a database: rejected as unnecessary for the initial local single-run scope.

## Decision: Parse ranked retrieval indices from `references` events

**Rationale**: Each SmartQ `SearchResult` includes `chunk_index`; parse valid non-negative integer values in received order, de-duplicate repeats, and retain the derived ranked list. Records with final answers but no valid reference indices remain generation-scorable and retrieval-unscorable.

**Alternatives considered**:

- Infer chunks by matching reference text: rejected because duplicates and text changes make it non-deterministic.
- Treat missing references as an empty retrieval list: rejected because it turns absent evidence into an incorrect zero retrieval score.

## Decision: Support deterministic generation metrics alongside existing retrieval metrics

**Rationale**: `src/evals/metrics.py` already provides `precision`, `recall`, `mrr`, `ndcg`, and `map` at configurable cutoffs. Add normalized exact match and character-level F1 for answer-vs-reference comparison. These are deterministic, fast, and work for Chinese answers without another model service.

**Alternatives considered**:

- RAGAS/LLM-as-a-judge metrics: rejected for v1 because they add external model credentials, non-determinism, cost, and run-time dependencies.
- Exact match only: rejected because partially correct concise answers would be invisible.
