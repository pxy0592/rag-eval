# Repository Guidelines

## Project Structure & Module Organization

This Python 3.12 project creates RAG evaluation datasets from SmartQ knowledge bases and optional Wikipedia articles. The installable package is `src` (the Hatch wheel configuration is `packages = ["src"]`), so imports within it must be package-relative and the console entry point is `src.main:main`.

- `src/main.py` defines the Gradio workflow, source selection, and paginated SmartQ UI callbacks.
- `src/settings.py` reads model, OpenAI-compatible, and SmartQ environment settings.
- `src/core/generation.py` owns retrieval-context/history compression, prompt construction, and streaming answer generation.
- `src/lib/smartq.py` is the authenticated SmartQ REST client. It lists knowledge and chunks by page, then builds complete `Article` objects for Q/A generation.
- `src/lib/` also contains LLM/prompt code, shared Pydantic and TypedDict models, utilities, Wikipedia retrieval, embeddings/reranking, `vectordb.py` hybrid retrieval, and the SmartQ Agent QA and knowledge-chat SSE clients.
- `src/evals/` implements the local SmartQ evaluation CLI: validation-set collection through Agent QA or knowledge-base QA, append-only result records, deterministic retrieval/generation scoring, and JSON/Markdown reports.
- `src/lib/models/llm.py` uses vLLM 0.27's `StructuredOutputsParams`; do not reintroduce the removed `GuidedDecodingParams` API.
- `tests/` mirrors source responsibilities: application flow in `tests/test_main.py`, generation flow in `tests/core/`, and library tests in `tests/lib/`, including vector-store persistence and reranking behavior.
- `dataset/` holds JSONL data; `paper/` contains Typst source and visual assets. Treat generated data and paper artifacts as deliberate changes.

## Build, Test, and Development Commands

Use `uv` so dependencies match `uv.lock`:

```bash
uv sync                         # install runtime and development dependencies
uv run pytest                   # run the complete test suite
uv run pytest --cov=src         # run tests with coverage reporting
uv run python -m src.main       # launch the Gradio UI
```

Copy `.env.example` to `.env`. For development-model mode, set `ENVIRONMENT=dev`, `CLIENT_URL`, `OPENAI_API_KEY`, and `LLM_MODEL`. For SmartQ, set the **backend** base URL—not the documentation UI—and an API key:

```dotenv
SMARTQ_API_URL=http://localhost:8080
SMARTQ_API_KEY=sk-...
SMARTQ_AGENT_ID=builtin-quick-answer
SMARTQ_MODEL_ID=builtin-llm-qwen3-32b
```

Never commit `.env`, API keys, downloaded private knowledge content, test-generated vector-store files, or `evaluation_runs/` artifacts. `KnowledgeBase(test=True)` writes its temporary persistence fixture under the system temp directory (`/tmp/rag-eval` on Linux); production instances use the repository `data/` directory.

## Coding Style & Naming Conventions

Use 4-space indentation, clear type annotations, `snake_case` for functions/tests, `PascalCase` for models, and uppercase configuration names. Keep Gradio callbacks thin; put SmartQ HTTP and pagination logic in `src/lib/smartq.py`. Preserve SmartQ’s `X-API-Key` header and its `data`/`total` pagination envelope. The UI page size is 20; do not silently change it without updating tests and labels.

Use package-relative imports inside `src` (for example, `from .types import Chunk` or `from ..lib.settings import settings`), not top-level `lib` or `core` imports. Keep source-article and retrieval metadata compatible with the shared `Chunk`, `Document`, and `RetrievedChunk` models. For hybrid retrieval, preserve candidate-index mapping after reranking and keep reranker scores normalized before threshold filtering.

## Testing Guidelines

Use `pytest`, fixtures, parametrization, and mocks for model, network, embedding, reranker, and SmartQ API boundaries. Name tests `test_<behavior>`. Cover SmartQ page navigation, Chinese titles, chunk ordering, API errors, missing configuration, generation prompt/context handling, and vector insert/save/load/reranking behavior.

Mock API and model responses; tests must not require a live SmartQ server, GPU, external Wikipedia connection, or an actual vLLM/embedding-model initialization. SmartQ evaluation tests must mock session creation and both Agent QA and knowledge-chat SSE events, then assert source-order collection, retrieval-index handling, coverage counts, and saved-run re-scoring without a network call. Persistence tests must use `KnowledgeBase(test=True)` and must not write artifacts under the repository `data/` directory.

## Commit & Pull Request Guidelines

Use concise conventional prefixes such as `feat:`, `fix:`, `test:`, `refact:`, `build:`, or `docs:`. PRs should describe user-visible behavior, validation commands, configuration assumptions, and affected data artifacts. Include screenshots for visible Gradio changes and link related issues when available.
