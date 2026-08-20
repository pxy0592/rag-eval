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


def test_generate_uses_qa_pair_already_in_selected_chunk(main_module, monkeypatch):
    article = Article(
        title="已有问答",
        source="http://smartq/api/v1/knowledge/id-1",
        language="cn",
        chunks=[
            Chunk(
                heading="分块 1",
                level=1,
                content="问题：截至2014年，该单位共有多少人？\n答案：4797人。",
            )
        ],
    )
    generate = Mock(
        return_value=SimpleNamespace(
            question="should not replace original question",
            answer="截至2014年，该单位共有4797人。",
        )
    )
    monkeypatch.setattr(main_module, "generate", generate)

    question, answer = main_module.generate_syntetic_qa_pair(
        "factual", article, [0]
    )

    assert question["value"] == "截至2014年，该单位共有多少人？"
    assert answer["value"] == "截至2014年，该单位共有4797人。"
    generate.assert_called_once()
    assert "Current answer: 4797人。" in generate.call_args.kwargs["prompt"]


def test_generate_and_add_qa(main_module, article, monkeypatch):
    generated = SimpleNamespace(
        question="What was generated?",
        answer="The model generated a complete factual answer.",
    )
    generate = Mock(return_value=generated)
    monkeypatch.setattr(main_module, "generate", generate)

    question, answer = main_module.generate_syntetic_qa_pair("factual", article, [0])

    assert question["value"] == "What was generated?"
    assert answer["value"] == "The model generated a complete factual answer."
    generate.assert_called_once()
    assert "Source title: Test article" in generate.call_args.kwargs["prompt"]

    qa_data = main_module.add_to_qa_dataset(
        "factual", "en", "Test article", [0], "Q", "A", []
    )
    assert len(qa_data) == 1
    assert isinstance(qa_data[0], QA)
    assert qa_data[0].question == "Q"


def test_short_generated_answer_gets_one_contextual_rewrite(
    main_module, monkeypatch
):
    article = Article(
        title="锦州机务段",
        source="http://smartq/api/v1/knowledge/id-1",
        language="cn",
        chunks=[
            Chunk(
                heading="分块 1",
                level=1,
                content="截至2020年末，全段现员4041人。",
            )
        ],
    )
    generate = Mock(
        side_effect=[
            SimpleNamespace(
                question="截至2020年末，全段共有多少名现员？",
                answer="4041人",
            ),
            SimpleNamespace(
                question="ignored rewrite question",
                answer="截至2020年末，锦州机务段共有4041名现员。",
            ),
        ]
    )
    monkeypatch.setattr(main_module, "generate", generate)

    question, answer = main_module.generate_syntetic_qa_pair(
        "factual", article, [0]
    )

    assert question["value"] == "截至2020年末，全段共有多少名现员？"
    assert answer["value"] == "截至2020年末，锦州机务段共有4041名现员。"
    assert generate.call_count == 2
    assert "Do not return only a bare number" in generate.call_args_list[0].kwargs[
        "prompt"
    ]
    assert "Current answer: 4041人" in generate.call_args_list[1].kwargs[
        "prompt"
    ]


def test_answer_expansion_detects_fragments_but_not_complete_sentences(main_module):
    assert main_module.answer_needs_expansion("4041人") is True
    assert main_module.answer_needs_expansion("Paris.") is True
    assert (
        main_module.answer_needs_expansion(
            "截至2020年末，锦州机务段共有4041名现员。"
        )
        is False
    )
    assert (
        main_module.answer_needs_expansion(
            "Paris is the capital of France."
        )
        is False
    )


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


def test_parse_bulk_document_ids_supports_commas_newlines_and_dedup(
    main_module,
):
    assert main_module.parse_smartq_document_ids(
        "doc-1, doc-2\ndoc-1\r\ndoc-3"
    ) == ["doc-1", "doc-2", "doc-3"]


def test_random_chunk_selection_excludes_edges_and_is_non_adjacent(main_module):
    selected = main_module.select_random_non_adjacent_chunks(
        total_chunks=40,
        count=6,
        rng=main_module.random.Random(7),
    )

    assert len(selected) == 6
    assert all(10 <= index < 30 for index in selected)
    assert all(right - left > 1 for left, right in zip(selected, selected[1:]))


def test_random_chunk_selection_rejects_impossible_count(main_module):
    with pytest.raises(ValueError, match="at most 3 non-adjacent chunks"):
        main_module.select_random_non_adjacent_chunks(
            total_chunks=25,
            count=4,
            rng=main_module.random.Random(1),
        )


def test_bulk_generation_uses_floor_count_one_chunk_per_pair(
    main_module, monkeypatch
):
    def article(title):
        return Article(
            title=title,
            source=f"http://smartq/api/v1/knowledge/{title}",
            language="cn",
            chunks=[
                Chunk(heading=f"分块 {index + 1}", level=1, content=f"内容 {index}")
                for index in range(40)
            ],
        )

    class FakeSmartQClient:
        def get_article(self, document_id):
            return article(document_id)

    monkeypatch.setattr(
        main_module,
        "create_synthetic_qa_pair",
        lambda type_q, article, chunks_idx: SimpleNamespace(
            question=f"{article.title} question {chunks_idx[0]}",
            answer=f"{article.title} answer {chunks_idx[0]}",
        ),
    )

    qa_pairs, status = main_module.generate_bulk_smartq_qa(
        "doc-1\ndoc-2",
        5,
        client=FakeSmartQClient(),
        rng=main_module.random.Random(3),
    )

    assert len(qa_pairs) == 4
    assert status == (
        "Generated 4 Q/A pairs from 2 documents (2 per document). "
        "Requested total 5 leaves a remainder of 1 after integer division."
    )
    assert {qa.article_title for qa in qa_pairs} == {"doc-1", "doc-2"}
    for title in ("doc-1", "doc-2"):
        indexes = [
            qa.chunks[0] for qa in qa_pairs if qa.article_title == title
        ]
        assert len(indexes) == 2
        assert all(10 <= index < 30 for index in indexes)
        assert abs(indexes[0] - indexes[1]) > 1
        assert all(len(qa.chunks) == 1 for qa in qa_pairs)


def test_bulk_generation_requires_total_at_least_document_count(main_module):
    with pytest.raises(ValueError, match="at least the number of document IDs"):
        main_module.generate_bulk_smartq_qa(
            "doc-1,doc-2",
            1,
            client=Mock(),
        )
