import importlib
import os
from types import SimpleNamespace
from unittest.mock import Mock, patch

for proxy_name in ("ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "http_proxy", "https_proxy"):
    os.environ.pop(proxy_name, None)

import gradio as gr
import pytest

from src.lib.smartq import SmartQChunk, SmartQChunkPage, SmartQKnowledge, SmartQKnowledgePage
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


def test_smartq_knowledge_and_chunk_page_controls(main_module, monkeypatch):
    knowledge_page = SmartQKnowledgePage(
        items=[SmartQKnowledge(id="knowledge-21", title="第二十一篇知识")],
        page=2,
        page_size=20,
        total=21,
    )
    chunk_page = SmartQChunkPage(
        items=[SmartQChunk(index=20, content="第二十一段")],
        page=2,
        page_size=20,
        total=41,
    )
    monkeypatch.setattr(
        main_module, "get_smartq_knowledge_page", Mock(return_value=knowledge_page)
    )
    monkeypatch.setattr(
        main_module, "get_smartq_chunk_page", Mock(return_value=chunk_page)
    )

    knowledge_updates = main_module.load_smartq_knowledge_page("kb-1")
    chunk_updates = main_module.next_smartq_chunk_page("knowledge-1", 1)

    assert knowledge_updates[0]["choices"] == [("第二十一篇知识", "knowledge-21")]
    assert knowledge_updates[1:4] == ("kb-1", 2, "Knowledges: 21-21 of 21 (page 2/2)")
    assert knowledge_updates[4]["interactive"] is True
    assert knowledge_updates[5]["interactive"] is False
    main_module.get_smartq_knowledge_page.assert_called_once_with("kb-1", 1)

    assert chunk_updates[0] == [[21, "第二十一段"]]
    assert chunk_updates[1:3] == (2, "Chunks: 21-21 of 41 (page 2/3)")
    assert chunk_updates[3]["interactive"] is True
    assert chunk_updates[4]["interactive"] is True
    main_module.get_smartq_chunk_page.assert_called_once_with("knowledge-1", 2)


def test_smartq_page_navigation_never_requests_page_zero(main_module, monkeypatch):
    knowledge_page = SmartQKnowledgePage(items=[], page=1, page_size=20, total=0)
    monkeypatch.setattr(
        main_module, "get_smartq_knowledge_page", Mock(return_value=knowledge_page)
    )

    main_module.previous_smartq_knowledge_page("kb-1", 1)

    main_module.get_smartq_knowledge_page.assert_called_once_with("kb-1", 1)


def test_dataset_source_name_uses_matching_article_source(main_module):
    smartq_articles = [
        {"title": "中文知识", "source": "http://smartq/api/v1/knowledge/id-1"}
    ]
    wikipedia_articles = [
        {"title": "Prime number", "source": "https://en.wikipedia.org/?curid=1"}
    ]

    assert main_module.dataset_source_name(smartq_articles) == "smartq"
    assert main_module.dataset_source_name(wikipedia_articles) == "wikipedia"
    assert (
        main_module.dataset_source_name(
            smartq_articles,
            [{"article_title": "中文知识"}],
        )
        == "smartq"
    )
    assert main_module.dataset_source_name(smartq_articles + wikipedia_articles) == "dataset"


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
