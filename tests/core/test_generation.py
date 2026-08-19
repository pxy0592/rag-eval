from unittest.mock import MagicMock

import pytest

from src.core.generation import (
    compress_context,
    compress_history,
    create_prompt,
    generate_answer,
)
from src.lib.types import Chunk, RetrievedChunk

# CONSTANTS --------------------------------------------------------------------

MAX_TOKENS = 100  # Simulated max token limit


# Fixtures ---------------------------------------------------------------------


@pytest.fixture
def chat_history():
    return [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "What's the weather today?"},
    ]


@pytest.fixture
def chunks():
    return [
        RetrievedChunk(
            chunk=Chunk(id="1", doc_id="1", page=12, text="Content A"),
            scores={
                "dense_score": 1.0,
                "sparse_score": 1.0,
                "colbert_score": 1.0,
                "hybrid_score": 1.0,
                "rerank_score": 1.0,
            },
        ),
        RetrievedChunk(
            chunk=Chunk(id="2", doc_id="1", page=13, text="Content B"),
            scores={
                "dense_score": 1.0,
                "sparse_score": 1.0,
                "colbert_score": 1.0,
                "hybrid_score": 1.0,
                "rerank_score": 1.0,
            },
        ),
    ]


# Test Cases -------------------------------------------------------------------


def test_compress_history_under_limit(chat_history):
    result = compress_history(chat_history, max_tokens=MAX_TOKENS)
    assert isinstance(result, list)
    assert len(result) == len(chat_history)


def test_compress_history_truncates():
    history = [{"role": "user", "content": "A" * 400}] * 5
    result = compress_history(history, max_tokens=100)
    assert len(result) < len(history)


def test_compress_context_empty():
    assert compress_context([], max_tokens=MAX_TOKENS) == ""


def test_compress_context_content(chunks):
    result = compress_context(chunks, max_tokens=MAX_TOKENS)
    assert "[Doc 1]" in result and "Content A" in result
    assert isinstance(result, str)
    assert len(result) <= MAX_TOKENS * 4


def test_create_prompt_with_context(chat_history):
    query = "What is AI?"
    context = "Context goes here"
    result = create_prompt(query, chat_history, context)
    assert "<|User|>" in result and "<|Assistant|>" in result
    assert "Given the context" in result


def test_create_prompt_without_context(chat_history):
    query = "What is AI?"
    result = create_prompt(query, chat_history, context="")
    assert "<|Assistant|> <think>" in result


def test_generate_answer(monkeypatch, chat_history, chunks):
    mock_model = MagicMock()
    mock_model.generate_stream.return_value = iter([
        "Step 1",
        "Step 2",
        "Answer",
    ])

    generator = generate_answer(
        query="What is AI?",
        history=chat_history,
        chunks=chunks,
        model=mock_model,
        params={},
    )

    outputs = list(generator)
    assert outputs == ["Step 1", "Step 2", "Answer"]
