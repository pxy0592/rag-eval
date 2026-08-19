# Repository Guidelines

## Project Structure & Module Organization

This Python 3.12 project creates RAG evaluation datasets from SmartQ knowledge bases and optional Wikipedia articles.

- `src/main.py` defines the Gradio workflow, source selection, and paginated SmartQ UI callbacks.
- `src/settings.py` reads model, OpenAI-compatible, and SmartQ environment settings.
- `src/lib/smartq.py` is the authenticated SmartQ REST client. It lists knowledge and chunks by page, then builds complete `Article` objects for Q/A generation.
- `src/lib/` also contains LLM/prompt code, Pydantic models, utilities, and Wikipedia retrieval.
- `tests/` mirrors source responsibilities: application flow in `tests/test_main.py`, library tests in `tests/lib/`.
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
```

Never commit `.env`, API keys, or downloaded private knowledge content.

## Coding Style & Naming Conventions

Use 4-space indentation, clear type annotations, `snake_case` for functions/tests, `PascalCase` for models, and uppercase configuration names. Keep Gradio callbacks thin; put SmartQ HTTP and pagination logic in `src/lib/smartq.py`. Preserve SmartQ’s `X-API-Key` header and its `data`/`total` pagination envelope. The UI page size is 20; do not silently change it without updating tests and labels.

## Testing Guidelines

Use `pytest`, fixtures, parametrization, and mocks for model, network, and SmartQ API boundaries. Name tests `test_<behavior>`. Cover SmartQ page navigation, Chinese titles, chunk ordering, API errors, and missing configuration. Mock API responses; tests must not require a live SmartQ server, GPU, or external Wikipedia connection.

## Commit & Pull Request Guidelines

Use concise conventional prefixes such as `feat:`, `fix:`, `test:`, `refact:`, `build:`, or `docs:`. PRs should describe user-visible behavior, validation commands, configuration assumptions, and affected data artifacts. Include screenshots for visible Gradio changes and link related issues when available.
