# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python 3.12 WikiQA dataset generator for RAG evaluation.

- `src/main.py` defines the Gradio workflow and application entry point.
- `src/settings.py` loads environment-based runtime settings.
- `src/lib/` contains reusable modules: LLM clients and prompting (`llm.py`, `prompt.py`), Pydantic data models (`types.py`), text/file helpers (`utils.py`), and Wikipedia retrieval/chunking (`wikipedia.py`).
- `tests/` mirrors the source responsibilities, with shared application tests in `tests/test_main.py` and module tests under `tests/lib/`.
- `dataset/` stores JSONL input/output data; `paper/` contains the Typst paper and visual assets. Treat generated datasets and paper artifacts as deliberate changes.

## Build, Test, and Development Commands

Use `uv` to keep dependencies aligned with `uv.lock`:

```bash
uv sync                         # install runtime and development dependencies
uv run pytest                   # run the complete test suite
uv run pytest --cov=src         # run tests with coverage reporting
uv run python -m src.main       # launch the local Gradio UI
```

Copy `.env.example` to `.env` before running the UI and configure `LLM_MODEL`, `DTYPE`, `ENVIRONMENT` (`dev` or `prod`), and `CLIENT_URL`. Never commit `.env` or credentials.

## Coding Style & Naming Conventions

Follow standard Python style with 4-space indentation, readable line lengths, and type annotations for public functions. Use `snake_case` for functions, variables, and test names; `PascalCase` for classes and Pydantic models; and uppercase names for configuration fields. Keep UI callbacks thin and place reusable behavior in `src/lib/`. No repository-wide formatter or linter is configured, so review imports, formatting, and dead code before committing.

## Testing Guidelines

Tests use `pytest`, fixtures, parametrization, and mocks for network and LLM boundaries. Name files `test_*.py` and tests `test_<behavior>`. Add or update focused unit tests whenever changing parsing, chunking, model validation, configuration, or UI callback behavior. Keep tests deterministic: mock Wikipedia requests and model clients rather than requiring network access or GPUs.

## Commit & Pull Request Guidelines

Use concise, imperative, conventional-style prefixes such as `feat:`, `fix:`, `test:`, `refact:`, `build:`, or `docs:`; explain the user-visible or maintenance intent. Pull requests should describe the change, validation commands, configuration assumptions, and any dataset or paper artifacts affected. Include screenshots for visible Gradio UI changes and link related issues when applicable.
