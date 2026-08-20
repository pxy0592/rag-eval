# Feature Specification: SmartQ Agent Evaluation

**Feature Branch**: `001-smartq-agent-evaluation`  
**Created**: 2026-08-19  
**Status**: Draft  
**Input**: User description: "Use `dataset/smartq_qa_1.json` to query the SmartQ Agent QA interface, save per-question results, calculate retrieval and generation scores against the validation answers, and produce a report."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Run a reproducible Agent validation set (Priority: P1)

An evaluation engineer selects a Q/A validation-set JSON file and starts an evaluation run. The system sends every selected question to the configured SmartQ Agent QA service one at a time, records the returned answer and retrieval evidence, and saves a durable per-question result record so the run can be inspected or scored later.

**Why this priority**: Without a complete, auditable collection run, no retrieval or generation quality can be measured.

**Independent Test**: With a fixture validation set and a mocked Agent QA service, run the collector and verify that requests follow the source order, only one request is active at a time, and one persisted record is produced for every source question, including failures.

**Acceptance Scenarios**:

1. **Given** a valid validation-set JSON file and configured Agent QA credentials, **When** an engineer starts a full collection run, **Then** every question is submitted exactly once in source order with no concurrent Agent QA requests.
2. **Given** an Agent QA response contains an answer and retrieval references, **When** the question completes, **Then** the persisted result retains the source question, reference answer, expected chunk identifiers, generated answer, returned retrieval identifiers, and response diagnostics needed for later scoring.
3. **Given** an Agent QA request fails or returns an unusable response, **When** collection continues, **Then** the system records that question as failed with an actionable error and continues with the remaining questions instead of silently omitting it.

---

### User Story 2 - Measure retrieval and generation quality (Priority: P1)

After collection, an evaluation engineer calculates all supported metrics or an explicitly selected subset against the validation-set ground truth and receives separate retrieval and generation results, including the number of scored, failed, and unscorable questions.

**Why this priority**: The primary value of the feature is a trustworthy quality measurement, not merely storing Agent answers.

**Independent Test**: With deterministic saved results that include reference answers and returned chunk identifiers, run scoring and verify the expected values for every selected metric and the correct handling of missing or failed records.

**Acceptance Scenarios**:

1. **Given** completed result records with retrieval identifiers, **When** retrieval scoring runs, **Then** it compares their ranked identifiers with the validation-set chunk identifiers and calculates the retrieval metrics defined by the project.
2. **Given** completed result records with generated and reference answers, **When** generation scoring runs, **Then** it calculates every supported requested generation metric using the project-defined comparison rules.
3. **Given** a result lacks retrieval evidence, **When** retrieval scoring runs, **Then** the report marks that result unscorable for retrieval and never substitutes an answer-quality score for a retrieval score.
4. **Given** the engineer selects a subset of supported metrics, **When** scoring runs, **Then** the report contains only that subset and clearly identifies omitted and unavailable metrics.

---

### User Story 3 - Review and share a quality report (Priority: P2)

An evaluation engineer generates a human-readable and machine-readable report for a saved run, with a concise overall summary and per-question evidence that explains failures and unscorable results.

**Why this priority**: A score without traceability cannot be used to improve the SmartQ system or compare later runs.

**Independent Test**: With a saved run containing successful, failed, and unscorable records, generate a report and verify its totals, requested metrics, per-question diagnostics, and output locations.

**Acceptance Scenarios**:

1. **Given** a saved evaluation run, **When** reporting completes, **Then** it writes a machine-readable metrics artifact and a readable summary containing the dataset identity, run time, selected metrics, aggregate scores, and coverage counts.
2. **Given** any failed or unscorable questions, **When** reporting completes, **Then** the report identifies their source question and reason without exposing API keys or private credentials.

### Edge Cases

- An input JSON file is empty, malformed, or has invalid Q/A records: fail validation before contacting SmartQ and describe the invalid location.
- A selected metric is unsupported: reject the request with the list of supported metric names.
- A response returns an answer but no usable chunk/reference identifiers: retain the answer for generation scoring and mark retrieval scoring unavailable for that record.
- A response contains duplicate, non-numeric, or unordered retrieval identifiers: normalize only valid identifiers while preserving their returned ranking; record discarded values in diagnostics.
- The output directory already contains a run with the same identifier: do not overwrite it unless the engineer explicitly requests replacement.
- A previously interrupted run is re-evaluated: existing records remain auditable; the system must not mistake partial data for a completed run.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST load a validation-set JSON array whose records contain a question, reference answer, and expected chunk identifier list, and must preserve source-record order.
- **FR-002**: The system MUST support `dataset/smartq_qa_1.json` as a valid input without requiring modification of the dataset.
- **FR-003**: The system MUST send selected questions to the configured SmartQ Agent QA interface sequentially, with at most one in-flight request per run.
- **FR-004**: The system MUST obtain SmartQ endpoint, API authentication, and Agent selection from configuration rather than source code or dataset records.
- **FR-005**: The system MUST persist one durable per-question result record before a run is considered complete; each record must include run identity, source-record identity, request question, reference answer, expected chunk identifiers, Agent answer when available, returned retrieval identifiers when available, status, timing, and error/diagnostic details.
- **FR-006**: The system MUST continue collecting remaining questions after an individual Agent request failure and MUST report failures separately from successful results.
- **FR-007**: The system MUST calculate the project-defined retrieval metrics from returned ranked retrieval identifiers and validation-set chunk identifiers for every retrieval-scorable successful result.
- **FR-008**: The system MUST calculate the project-defined generation metrics from each generated answer and its validation-set reference answer for every generation-scorable successful result.
- **FR-009**: The system MUST allow callers to request all supported metrics or a named subset and MUST reject unsupported metric names before scoring.
- **FR-010**: The system MUST produce both a machine-readable metrics artifact and a human-readable report containing aggregate scores, selected metric names, denominator/coverage counts, failed-record counts, unscorable-record counts, and per-question diagnostic references.
- **FR-011**: The system MUST keep raw Agent response payloads and result artifacts local, must not store credentials in them, and must avoid logging credentials.
- **FR-012**: The system MUST expose a runnable interface that supports collection, scoring, and reporting as one end-to-end evaluation operation and supports scoring/reporting an already saved run without recalling the Agent.

### Key Entities

- **Validation Record**: A source Q/A item containing its source position, question, reference answer, expected chunk identifiers, language, type, and article title.
- **Evaluation Run**: The uniquely identified local evaluation attempt, its input dataset identity, selected metrics, configuration-safe Agent identity, timestamps, and completion status.
- **Agent Result Record**: The durable outcome for one validation record, including request/response evidence, returned answer, returned ranked retrieval identifiers, status, timing, and diagnostics.
- **Metric Result**: A named aggregate score together with its scoreable, failed, and unscorable denominators.
- **Evaluation Report**: The machine-readable metrics summary and readable run report produced from one saved evaluation run.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A valid 3-question validation set can be collected into exactly 3 durable records in the same source order, with no more than one Agent request active at any time.
- **SC-002**: For a deterministic fixture run, every requested retrieval and generation metric matches its documented expected value, and every aggregate score states its scoring denominator.
- **SC-003**: A completed run produces both report formats and lets an engineer identify the result status and scoring eligibility for 100% of source questions without re-querying SmartQ.
- **SC-004**: When an individual Agent request fails, the final report still accounts for 100% of input questions as successful, failed, or unscorable.
- **SC-005**: Re-scoring a saved completed run produces the same metrics result when the selected metric set and saved records are unchanged.

## Assumptions

- The configured SmartQ Agent QA response provides a final answer and, when retrieval metrics are expected, enough source/reference metadata to derive ranked validation-compatible chunk identifiers.
- Agent credentials and a target Agent identifier are supplied through environment configuration and are available only in the execution environment.
- The first release uses local JSON/JSONL artifacts for raw results and reports; remote result storage and a UI are out of scope.
- Generation metrics are deterministic comparison metrics defined in the project; model-judge metrics requiring another external model are out of scope unless later added explicitly.
- Full dataset collection is sequential by design to make runs reproducible and avoid overwhelming the target Agent service.
