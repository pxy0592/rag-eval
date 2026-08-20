import json
from unittest.mock import Mock

import pytest

from src.lib.smartq import SmartQAPIError, SmartQClient


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_list_knowledge_page_preserves_chinese_titles_and_page_metadata(monkeypatch):
    urlopen = Mock(
        return_value=FakeResponse(
            {
                "success": True,
                "total": 21,
                "page": 2,
                "page_size": 20,
                "data": [{"id": "knowledge-21", "title": "第二十一篇知识"}],
            }
        )
    )
    monkeypatch.setattr("src.lib.smartq.urlopen", urlopen)
    client = SmartQClient("http://smartq.example", "secret-key")

    result = client.list_knowledge_page("kb-1", page=2)

    assert [(knowledge.id, knowledge.title) for knowledge in result.items] == [
        ("knowledge-21", "第二十一篇知识")
    ]
    assert result.total == 21
    assert result.has_previous is True
    assert result.has_next is False
    request = urlopen.call_args.args[0]
    assert request.full_url.endswith(
        "/api/v1/knowledge-bases/kb-1/knowledge?page=2&page_size=20"
    )
    assert request.get_header("X-api-key") == "secret-key"


def test_get_chunk_page_uses_server_pagination(monkeypatch):
    urlopen = Mock(
        return_value=FakeResponse(
            {
                "success": True,
                "total": 41,
                "page": 2,
                "page_size": 20,
                "data": [
                    {"chunk_index": 20, "content": "第二十一段"},
                    {"chunk_index": 21, "content": "第二十二段"},
                ],
            }
        )
    )
    monkeypatch.setattr("src.lib.smartq.urlopen", urlopen)
    client = SmartQClient("http://smartq.example", "secret-key")

    result = client.get_chunk_page("knowledge-1", page=2)

    assert [(chunk.index, chunk.content) for chunk in result.items] == [
        (20, "第二十一段"),
        (21, "第二十二段"),
    ]
    assert result.has_previous is True
    assert result.has_next is True
    assert urlopen.call_args.args[0].full_url.endswith(
        "/api/v1/chunks/knowledge-1?page=2&page_size=20"
    )


def test_get_article_uses_all_processed_chunks_in_chinese_order(monkeypatch):
    urlopen = Mock(
        side_effect=[
            FakeResponse(
                {
                    "success": True,
                    "data": {"title": "设备检修规程", "description": "中文资料"},
                }
            ),
            FakeResponse(
                {
                    "success": True,
                    "total": 2,
                    "data": [
                        {"chunk_index": 1, "content": "第二段内容"},
                        {"chunk_index": 0, "content": "第一段内容"},
                    ],
                }
            ),
        ]
    )
    monkeypatch.setattr("src.lib.smartq.urlopen", urlopen)
    client = SmartQClient("http://smartq.example/api/v1/", "secret-key")

    article = client.get_article("knowledge-1")

    assert article.title == "设备检修规程"
    assert article.language == "cn"
    assert article.summary == "中文资料"
    assert [chunk.heading for chunk in article.chunks] == ["分块 1", "分块 2"]
    assert [chunk.content for chunk in article.chunks] == ["第一段内容", "第二段内容"]
    assert urlopen.call_args_list[0].args[0].full_url == (
        "http://smartq.example/api/v1/knowledge/knowledge-1"
    )
    assert urlopen.call_args_list[1].args[0].full_url.endswith(
        "/chunks/knowledge-1?page=1&page_size=100"
    )


@pytest.mark.parametrize(
    ("api_url", "api_key", "message"),
    [
        (None, "key", "SMARTQ_API_URL"),
        ("http://smartq.example", None, "SMARTQ_API_KEY"),
    ],
)
def test_client_requires_smartq_configuration(api_url, api_key, message):
    with pytest.raises(SmartQAPIError, match=message):
        SmartQClient(api_url, api_key)

class StreamingResponse:
    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def __iter__(self):
        return iter(self.lines)


def test_agent_qa_creates_session_parses_sse_and_preserves_rank(monkeypatch):
    from src.lib.smartq import SmartQAgentClient

    session_response = FakeResponse({"success": True, "data": {"id": "session-1"}})
    stream_response = StreamingResponse(
        [
            b'data: {"response_type":"answer","content":"404"}\n',
            b'data: {"response_type":"references","data":{"references":[{"chunk_index":51},{"chunk_index":51},{"chunk_index":108}]}}\n',
            'data: {"response_type":"answer","content":"1人"}\n'.encode("utf-8"),
            b'data: {"response_type":"complete"}\n',
        ]
    )
    urlopen = Mock(side_effect=[session_response, stream_response])
    monkeypatch.setattr("src.lib.smartq.urlopen", urlopen)

    client = SmartQAgentClient(
        "http://smartq.example",
        "secret-key",
        tenant_id="tenant-1",
        agent_id="agent-1",
        knowledge_base_ids=["kb-1"],
    )

    response = client.ask("\u73b0\u5458\u591a\u5c11\u4eba？")

    assert response.answer == "4041\u4eba"
    assert response.retrieved_chunk_indices == [51, 108]
    assert response.error is None
    assert urlopen.call_args_list[0].args[0].full_url == "http://smartq.example/api/v1/sessions"
    request = urlopen.call_args_list[1].args[0]
    assert request.full_url == "http://smartq.example/api/v1/agent-chat/session-1"
    assert request.get_header("X-api-key") == "secret-key"
    assert request.get_header("X-tenant-id") == "tenant-1"
    assert json.loads(request.data.decode("utf-8")) == {
        "query": "\u73b0\u5458\u591a\u5c11\u4eba？",
        "agent_id": "agent-1",
        "agent_enabled": True,
        "knowledge_base_ids": ["kb-1"],
        "channel": "api",
    }


def test_agent_qa_returns_sanitized_error_event(monkeypatch):
    from src.lib.smartq import SmartQAgentClient

    urlopen = Mock(
        side_effect=[
            FakeResponse({"success": True, "data": {"id": "session-1"}}),
            StreamingResponse(
                [
                    b'data: {"response_type":"error","content":"agent unavailable"}\n',
                    b'data: {"response_type":"complete"}\n',
                ]
            ),
        ]
    )
    monkeypatch.setattr("src.lib.smartq.urlopen", urlopen)

    response = SmartQAgentClient(
        "http://smartq.example", "secret-key", "tenant-1", "agent-1"
    ).ask("question")

    assert response.answer == ""
    assert response.retrieved_chunk_indices is None
    assert response.error == "agent unavailable"
    assert "secret-key" not in response.error


def test_agent_qa_redacts_api_key_from_persistable_sse_diagnostics(monkeypatch):
    from src.lib.smartq import SmartQAgentClient

    urlopen = Mock(
        side_effect=[
            FakeResponse({"success": True, "data": {"id": "session-1"}}),
            StreamingResponse(
                [
                    b'data: {"response_type":"answer","content":"secret-key answer"}\n',
                    b'data: {"response_type":"references","data":{"debug":"secret-key","references":[{"chunk_index":51}]}}\n',
                    b'data: {"response_type":"complete"}\n',
                ]
            ),
        ]
    )
    monkeypatch.setattr("src.lib.smartq.urlopen", urlopen)

    response = SmartQAgentClient(
        "http://smartq.example", "secret-key", "tenant-1", "agent-1"
    ).ask("question")

    serialized_events = json.dumps(response.events)
    assert "secret-key" not in serialized_events
    assert "secret-key" not in response.answer
    assert "[REDACTED]" in serialized_events
    assert response.answer == "[REDACTED] answer"
