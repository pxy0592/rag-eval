from unittest.mock import Mock, patch

import pytest

from src.lib.wikipedia import Article, Chunk, WikipediaProcessor


@pytest.fixture
def mock_requests():
    with patch("requests.get") as mock_get:
        yield mock_get


@pytest.fixture
def processor():
    return WikipediaProcessor()


def test_get_page_id_success(processor, mock_requests):
    mock_response = Mock()
    mock_response.json.return_value = {"query": {"search": [{"pageid": 12345}]}}
    mock_requests.return_value = mock_response

    page_id = processor._get_page_id("Prime Numbers", "en")
    assert page_id == 12345
    mock_requests.assert_called_once()


def test_get_lang_link_success(processor, mock_requests):
    mock_response = Mock()
    mock_response.json.return_value = {
        "query": {"pages": {"12345": {"langlinks": [{"*": "Número primo"}]}}}
    }
    mock_requests.return_value = mock_response

    translated_title = processor._get_lang_link(12345, "es")
    assert translated_title == "Número primo"


def test_parse_sections():
    processor = WikipediaProcessor()
    wikitext = """
    ==Introduction==
    A prime number is a natural number...
    
    ===History===
    The concept of prime numbers...
    
    Some more text here.
    """

    sections = processor._parse_sections(wikitext)
    assert len(sections) == 2
    assert sections[0].heading == "Introduction"
    assert sections[1].heading == "History"
    assert sections[1].level == 2


def test_split_text_paragraphs():
    processor = WikipediaProcessor()
    text = "Para 1\n\nPara 2\n\nPara 3"
    chunks = processor._split_text(text, 10)
    assert len(chunks) == 3
    assert chunks[0] == "Para 1"
    assert chunks[1] == "Para 2"


def test_split_text_sentences():
    processor = WikipediaProcessor()
    text = "First sentence. Second sentence. Third sentence."
    chunks = processor._split_text(text, 25)
    assert len(chunks) == 3
    assert "First sentence" in chunks[0]
    assert "Third sentence" in chunks[2]


def test_full_flow(processor, mock_requests):
    # Mock chain of responses
    mock_responses = [
        # _get_page_id (en)
        Mock(json=Mock(return_value={"query": {"search": [{"pageid": 123}]}})),
        # _get_lang_link (es)
        Mock(
            json=Mock(
                return_value={
                    "query": {"pages": {"123": {"langlinks": [{"*": "Número primo"}]}}}
                }
            )
        ),
        # _get_page_id (es)
        Mock(json=Mock(return_value={"query": {"search": [{"pageid": 456}]}})),
        # _get_page_content
        Mock(
            json=Mock(
                return_value={
                    "query": {
                        "pages": {
                            "456": {
                                "revisions": [
                                    {
                                        "slots": {
                                            "main": {
                                                "*": "==Introducción==\nTexto en español"
                                            }
                                        }
                                    }
                                ]
                            }
                        }
                    }
                }
            )
        ),
        # _get_page_title
        Mock(
            json=Mock(
                return_value={"query": {"pages": {"456": {"title": "Número primo"}}}}
            )
        ),
    ]
    mock_requests.side_effect = mock_responses

    article = processor.get_article("Prime Numbers", "es", 500)

    assert isinstance(article, Article)
    assert article.language == "es"
    assert len(article.chunks) > 0
    assert "es.wikipedia.org" in article.source


def test_article_not_found(processor, mock_requests):
    mock_requests.return_value = Mock(json=Mock(return_value={"query": {}}))
    article = processor.get_article("NonExistentPage", "en")
    assert article is None


def test_invalid_lang_link(processor, mock_requests):
    mock_responses = [
        Mock(json=Mock(return_value={"query": {"search": [{"pageid": 123}]}})),
        Mock(json=Mock(return_value={"query": {"pages": {"123": {}}}})),
    ]
    mock_requests.side_effect = mock_responses

    article = processor.get_article("Prime Numbers", "fr")
    assert article is None


def test_chunk_model_validation():
    chunk = Chunk(heading="Introduction", level=1, content="Sample content")
    assert chunk.heading == "Introduction"
    assert isinstance(chunk.model_dump(), dict)


def test_article_model_validation():
    article = Article(
        title="Test",
        source="https://example.com",
        language="en",
        chunks=[Chunk(heading="Test", level=1, content="Content")],
    )
    assert article.title == "Test"
    assert len(article.chunks) == 1
