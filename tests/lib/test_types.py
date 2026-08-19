from src.lib.types import Article, Chunk, QA, QAFormat


def test_chunk_mapping_access_and_serialization():
    chunk = Chunk(heading="Intro", level=1, content="Text")

    assert chunk["heading"] == "Intro"
    assert chunk.model_dump() == {"heading": "Intro", "level": 1, "content": "Text"}


def test_article_and_qa_serialization_defaults():
    article = Article(title="Test", source="url", language="en", chunks=[])
    qa = QA(
        type="factual",
        language="en",
        article_title="Test",
        chunks=[0],
        question="Q",
        answer="A",
    )

    assert article.summary == ""
    assert article.to_json()["title"] == "Test"
    assert qa.to_json()["answer"] == "A"


def test_qa_format_validation():
    qa = QAFormat(question="What?", answer="This.")

    assert qa.question == "What?"
    assert "question" in qa.model_json_schema()["properties"]
