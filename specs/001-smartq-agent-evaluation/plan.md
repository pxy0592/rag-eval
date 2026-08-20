# Implementation Plan: SmartQ Agent Evaluation

**Branch**: `001-smartq-agent-evaluation` | **Date**: 2026-08-19 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/001-smartq-agent-evaluation/spec.md`

## Summary

Add a deterministic, local-file evaluation workflow for SmartQ Agent QA. The workflow loads Q/A JSON validation records such as `dataset/smartq_qa_1.json`, sends each question sequentially to SmartQ's session-backed SSE Agent QA endpoint, stores one sanitized JSONL result record per source question, then scores saved results against reference answers and expected `chunk_index` values. It exposes collection, scoring, reporting, and an end-to-end CLI, preserving the current `src/evals` metric utilities while adding deterministic generation metrics and explicit coverage accounting.

## Technical Context

**Language/Version**: Python 3.12  
**Primary Dependencies**: stdlib `urllib`/`json` for SmartQ HTTP/SSE, Pydantic 2.13, pytest 9.1  
**Storage**: Local UTF-8 JSONL records plus JSON and Markdown reports under an ignored `evaluation_runs/<run-id>/` directory  
**Testing**: pytest with mocked `urlopen`, deterministic fixture data, and no live SmartQ, GPU, model, or internet requirement  
**Target Platform**: Linux/macOS/Windows command-line environment with Python 3.12 and SmartQ network access at runtime  
**Project Type**: Python package and CLI-oriented evaluation utility  
**Performance Goals**: Process input in source order with exactly one active Agent QA request; write each finished record immediately so partial runs remain auditable  
**Constraints**: Never put credentials in source, artifacts, or logs; streaming responses must be parsed incrementally; answer and retrieval scoring must be independently eligible; re-scoring must not contact SmartQ  
**Scale/Scope**: v1 evaluates JSON arrays like `smartq_qa_1.json`, one configured SmartQ Agent and optional knowledge-base scope per run; no UI, remote artifact store, concurrent collection, or LLM-as-a-judge metric

## Constitution Check

The project constitution is an unfilled template. No enforceable MUST/SHOULD principles apply; this check is skipped. Repository `AGENTS.md` constraints still apply: use `uv`, preserve `X-API-Key`, keep SmartQ HTTP logic in `src/lib/smartq.py`, and mock all network/model boundaries in tests.

**Pre-design gate**: PASS — no applicable constitution violation or unresolved technical clarification.

## Project Structure

### Documentation (this feature)

```text
specs/001-smartq-agent-evaluation/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── evaluation-cli.md
└── tasks.md
```

### Source Code (repository root)

```text
src/
├── settings.py                         # SmartQ Agent evaluation configuration
├── lib/
│   └── smartq.py                       # session creation and sequential Agent QA SSE client
└── evals/
    ├── __init__.py
    ├── models.py                       # validation/run/result/report Pydantic models
    ├── runner.py                       # load, collect, persist, score, report orchestration
    ├── cli.py                          # collect, score, report, and run commands
    └── metrics.py                      # existing retrieval metrics plus deterministic generation metrics

tests/
├── lib/test_smartq.py                  # SmartQ Agent session/SSE request and event parsing tests
└── evals/
    ├── test_metrics.py                 # retrieval and generation metric unit tests
    ├── test_runner.py                  # collection, persistence, scoring, and report tests
    └── test_cli.py                     # CLI contract tests

.env.example                            # documented SmartQ Agent configuration
.gitignore                              # excludes local evaluation artifacts
```

**Structure Decision**: Keep the existing package layout. The Agent QA protocol remains in `src/lib/smartq.py`, while evaluation-domain models and orchestration live in a dedicated `src/evals` package. This prevents the current experimental `collect.py`/`evaluate.py` scripts from owning transport or persistence policy.

## Complexity Tracking

No constitution violation requires a complexity exception.

## Post-Design Constitution Check

PASS — the design uses the existing package, a single local CLI, deterministic file artifacts, mocked boundaries, and no new runtime service or model dependency.
