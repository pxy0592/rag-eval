import importlib
import os
from types import SimpleNamespace
from unittest.mock import Mock, patch

for proxy_name in ("ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "http_proxy", "https_proxy"):
    os.environ.pop(proxy_name, None)

import gradio as gr
import pytest

from src.lib.types import Article, Chunk, QA


@pytest.fixture(scope="module")
def main_module():
    with patch("src.lib.llm.get_client", return_value=Mock()):
        return importlib.import_module("src.main")


@pytest.fixture
def article():
    return Article(
        title="Test article",
        source="https://example.com",
        language="en",
        chunks=[Chunk(heading="Intro", level=1, content="Useful context")],
    )


def test_get_articles_success_and_errors(main_module, article, monkeypatch, capsys):
    fetch = Mock(return_value=[article])
    monkeypatch.setattr(main_module, "get_wikipedia_article", fetch)

    assert main_module.get_articles("Wikipedia", "Test", ["en"]) == [article]
    fetch.assert_called_once_with("Test", ["en"])

    fetch.side_effect = ConnectionError("offline")
    assert main_module.get_articles("Wikipedia", "Test", ["en"]) == []
    assert "Network error" in capsys.readouterr().out

    fetch.side_effect = RuntimeError("broken")
    assert main_module.get_articles("Wikipedia", "Test", ["en"]) == []
    assert "Unexpected error" in capsys.readouterr().out


def test_get_articles_rejects_unknown_source(main_module, capsys):
    assert main_module.get_articles("Other", "Test", ["en"]) == []
    assert "Unsupported source" in capsys.readouterr().out


def test_generate_and_add_qa(main_module, article, monkeypatch):
    generated = SimpleNamespace(question="Generated?", answer="Generated.")
    monkeypatch.setattr(main_module, "generate", Mock(return_value=generated))

    question, answer = main_module.generate_syntetic_qa_pair("factual", article, [0])

    assert question["value"] == "Generated?"
    assert answer["value"] == "Generated."

    qa_data = main_module.add_to_qa_dataset(
        "factual", "en", "Test article", [0], "Q", "A", []
    )
    assert len(qa_data) == 1
    assert isinstance(qa_data[0], QA)
    assert qa_data[0].question == "Q"


def test_generate_qa_errors_are_wrapped(main_module, article, monkeypatch):
    monkeypatch.setattr(main_module, "generate", Mock(side_effect=RuntimeError("no model")))

    with pytest.raises(gr.Error, match="no model"):
        main_module.generate_syntetic_qa_pair("factual", article, [0])


def test_ui_builders_and_launch(main_module, monkeypatch):
    with gr.Blocks() as demo:
        articles = gr.State([])
        qa_data = gr.State([])
        main_module.build_article_tab(articles)
        main_module.build_qa_tab(articles, qa_data)
        main_module.build_save_tab(articles, qa_data)

    assert demo is not None

    launch = Mock()
    monkeypatch.setattr(main_module.gr.Blocks, "launch", launch)
    main_module.launch()
    launch.assert_called_once()
