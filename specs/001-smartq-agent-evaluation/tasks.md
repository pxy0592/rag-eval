---

description: "Implementation tasks for SmartQ Agent QA evaluation"
---

# Tasks: SmartQ Agent Evaluation

**Input**: Design documents from `/specs/001-smartq-agent-evaluation/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/evaluation-cli.md`, and `quickstart.md`

**Tests**: Required. The repository guidelines require mocked tests for network and model boundaries; all Agent QA, persistence, metric, and CLI behavior must be tested without a live SmartQ service or GPU.

**Organization**: Tasks are grouped by user story so each increment can be implemented and tested independently.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel because it changes different files and has no unfinished dependency.
- **[Story]**: The user story this task serves.
- Every task includes an exact file path.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Establish safe configuration and artifact locations before evaluation logic is added.

- [X] T001 Update `.env.example` and `src/settings.py` with validated SmartQ Agent evaluation configuration: tenant ID, Agent ID, optional knowledge-base IDs, and timeout.
- [X] T002 [P] Add ignored local run-artifact paths to `.gitignore` and document their location in `AGENTS.md`.
- [X] T003 [P] Create `src/evals/__init__.py` and the CLI module skeleton in `src/evals/cli.py` with no import-time network or model initialization.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Define data contracts and shared metric selection that all user stories use.

**⚠️ CRITICAL**: Complete this phase before any user-story implementation.

- [X] T004 Create Pydantic validation, run, result, metric-summary, and report models plus safe run-ID/path helpers in `src/evals/models.py` per `data-model.md`.
- [X] T005 Add a metric registry, selected-metric validation, retrieval metric naming at supported cutoffs, and deterministic normalized-answer helpers in `src/evals/metrics.py`.
- [X] T006 Add reusable JSON/JSONL load, append-only write, and saved-run read helpers in `src/evals/runner.py` that reject malformed artifacts with actionable errors.

**Checkpoint**: Evaluation data and metric contracts are available without calling SmartQ.

---

## Phase 3: User Story 1 - Run a reproducible Agent validation set (Priority: P1) 🎯 MVP

**Goal**: Collect every validation question sequentially from the SmartQ Agent QA SSE endpoint and persist one terminal record per source item.

**Independent Test**: Run collection against mocked session and SSE responses for a fixture dataset; verify source order, one request at a time, records for successes and failures, and no credentials in artifacts.

### Tests for User Story 1

- [X] T007 [P] [US1] Add mocked SmartQ session creation, Agent QA request, SSE `answer`/`references`/`error`/`complete` parsing, and reference `chunk_index` ordering tests in `tests/lib/test_smartq.py`.
- [X] T008 [P] [US1] Add fixture-driven validation-set loading, sequential collection, append-only JSONL persistence, per-record failure continuation, and credential-redaction tests in `tests/evals/test_runner.py`.
- [X] T009 [P] [US1] Add `collect` command argument/configuration and exit-behavior tests in `tests/evals/test_cli.py`.

### Implementation for User Story 1

- [X] T010 [US1] Extend `src/lib/smartq.py` with a configuration-validated session-backed SmartQ Agent QA client that sends `X-API-Key`/`X-Tenant-ID`, parses SSE incrementally, and returns sanitized answer/reference diagnostics.
- [X] T011 [US1] Implement validation-set loading, sequential Agent collection, terminal result construction, and immediate `records.jsonl` persistence in `src/evals/runner.py` using `src/evals/models.py`.
- [X] T012 [US1] Implement the `collect` command in `src/evals/cli.py` with dataset/run/output options and clear preflight errors for invalid data or missing configuration.

**Checkpoint**: A mocked dataset can be collected into durable, ordered records without live SmartQ access.

---

## Phase 4: User Story 2 - Measure retrieval and generation quality (Priority: P1)

**Goal**: Score saved results using requested retrieval and deterministic generation metrics, with explicit eligibility and denominator accounting.

**Independent Test**: Score fixed saved JSONL records and assert every requested metric value, selected-metric validation, failure count, and retrieval-unscorable/generation-scorable distinction.

### Tests for User Story 2

- [X] T013 [P] [US2] Add precision, recall, MRR, NDCG, MAP, normalized answer exact-match, and character-F1 unit tests in `tests/evals/test_metrics.py`.
- [X] T014 [P] [US2] Add saved-run scoring tests for all metrics, metric subsets, failed records, missing references, no eligible denominator, and no SmartQ calls in `tests/evals/test_runner.py`.
- [X] T015 [P] [US2] Add `score` command tests for `all`, comma-separated metric subsets, and unsupported metric rejection in `tests/evals/test_cli.py`.

### Implementation for User Story 2

- [X] T016 [US2] Implement deterministic generation aggregate metrics and retain project retrieval metrics with explicit `@k` names in `src/evals/metrics.py`.
- [X] T017 [US2] Implement saved-record scoring, eligibility/coverage counting, metric-subset selection, and `metrics.json` output in `src/evals/runner.py`.
- [X] T018 [US2] Implement the `score` command in `src/evals/cli.py` so saved runs can be scored without SmartQ credentials or network calls.

**Checkpoint**: A saved run yields reproducible retrieval and generation metric summaries without re-querying SmartQ.

---

## Phase 5: User Story 3 - Review and share a quality report (Priority: P2)

**Goal**: Produce traceable JSON and Markdown reports and expose the complete collect-score-report path.

**Independent Test**: Generate reports from a mixed saved run and verify run metadata, aggregate score coverage, selected metrics, failed/unscorable diagnostics, stable output paths, and no secret leakage.

### Tests for User Story 3

- [X] T019 [P] [US3] Add machine-readable/Markdown report content and per-question diagnostic tests in `tests/evals/test_runner.py`.
- [X] T020 [P] [US3] Add `report` and end-to-end `run` command sequencing tests in `tests/evals/test_cli.py`.

### Implementation for User Story 3

- [X] T021 [US3] Implement `metrics.json` serialization and readable `report.md` rendering in `src/evals/runner.py` with aggregate denominators and failed/unscorable records.
- [X] T022 [US3] Implement `report` and end-to-end `run` commands in `src/evals/cli.py`, ensuring re-reporting/re-scoring only reads saved records and that existing run directories are not silently overwritten.

**Checkpoint**: A complete evaluation produces `records.jsonl`, `metrics.json`, and `report.md`; each score and exception is traceable to a saved source question.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Verify configuration, documentation, and whole-feature behavior.

- [X] T023 [P] Update `specs/001-smartq-agent-evaluation/quickstart.md`, `AGENTS.md`, and `.env.example` with final command names, metric names, artifact layout, and credential-safe operational guidance.
- [X] T024 Run and resolve `uv run pytest` for the complete suite, including `tests/lib/test_smartq.py` and `tests/evals/`.
- [X] T025 Run the mocked end-to-end example from `specs/001-smartq-agent-evaluation/quickstart.md` and verify no generated evaluation artifacts are staged for version control.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1** has no dependencies.
- **Phase 2** depends on Phase 1 and blocks every user story.
- **US1** depends on Phase 2 and delivers the collection MVP.
- **US2** depends on the saved-record model from Phase 2; it can be implemented with fixture records before US1's live transport is complete, but it must integrate after US1.
- **US3** depends on US1 and US2 artifacts.
- **Phase 6** depends on all user stories.

### User Story Dependencies

- **US1 (P1)**: independent collection slice after the foundational models exist.
- **US2 (P1)**: independent saved-run scoring slice after foundational models exist; integrates with US1's record schema.
- **US3 (P2)**: report/rendering slice that consumes US1 and US2 outputs.

### Parallel Opportunities

- T002 and T003 can run in parallel after T001's configuration field names are agreed.
- T007–T009 can run in parallel because they cover separate test files/concerns.
- T013–T015 can run in parallel after foundational model names are stable.
- T019 and T020 can run in parallel.

## Parallel Example: User Story 1

```text
Task: "Add mocked Agent QA SSE parsing tests in tests/lib/test_smartq.py"
Task: "Add collection/persistence tests in tests/evals/test_runner.py"
Task: "Add collect CLI tests in tests/evals/test_cli.py"
```

## Implementation Strategy

### MVP First

1. Finish Phases 1 and 2.
2. Finish US1 and demonstrate an ordered, durable mocked collection run.
3. Finish US2 and verify deterministic scores from saved fixture records.
4. Finish US3 and produce both final report formats.

### Incremental Delivery

- US1 provides audited raw SmartQ Agent results.
- US2 adds repeatable quality scores without another SmartQ call.
- US3 makes results reviewable and runnable through one CLI workflow.
