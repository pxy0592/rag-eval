"""Client for retrieving processed knowledge from a SmartQ knowledge base."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .types import Article, Chunk


class SmartQAPIError(RuntimeError):
    """Raised when SmartQ cannot return usable knowledge data."""


@dataclass(frozen=True)
class SmartQKnowledge:
    """Selectable metadata returned by the SmartQ knowledge list API."""

    id: str
    title: str
    description: str = ""


@dataclass(frozen=True)
class SmartQChunk:
    """A text chunk displayed in the paginated SmartQ chunk list."""

    index: int
    content: str


@dataclass(frozen=True)
class SmartQKnowledgePage:
    """One server-backed page of SmartQ knowledge records."""

    items: list[SmartQKnowledge]
    page: int
    page_size: int
    total: int

    @property
    def has_previous(self) -> bool:
        return self.page > 1

    @property
    def has_next(self) -> bool:
        return self.page * self.page_size < self.total


@dataclass(frozen=True)
class SmartQChunkPage:
    """One server-backed page of processed SmartQ text chunks."""

    items: list[SmartQChunk]
    page: int
    page_size: int
    total: int

    @property
    def has_previous(self) -> bool:
        return self.page > 1

    @property
    def has_next(self) -> bool:
        return self.page * self.page_size < self.total


class SmartQClient:
    """Read processed knowledge and text chunks through SmartQ's REST API."""

    _API_PATH = "/api/v1"
    DEFAULT_PAGE_SIZE = 20
    _ARTICLE_CHUNK_PAGE_SIZE = 100

    def __init__(self, api_url: str | None, api_key: str | None):
        if not api_url:
            raise SmartQAPIError("SMARTQ_API_URL must be set before using SmartQ")
        if not api_key:
            raise SmartQAPIError("SMARTQ_API_KEY must be set before using SmartQ")

        base_url = api_url.rstrip("/")
        self.api_url = (
            base_url
            if base_url.endswith(self._API_PATH)
            else f"{base_url}{self._API_PATH}"
        )
        self._api_key = api_key

    def list_knowledge_page(
        self, knowledge_base_id: str | None, page: int = 1, page_size: int = DEFAULT_PAGE_SIZE
    ) -> SmartQKnowledgePage:
        """Retrieve a single page of knowledge records from SmartQ."""

        knowledge_base_id = self._require_value(knowledge_base_id, "Knowledge base ID")
        payload = self._get_page(
            f"/knowledge-bases/{quote(knowledge_base_id, safe='')}/knowledge",
            page,
            page_size,
        )
        items = [
            SmartQKnowledge(
                id=knowledge_id,
                title=str(record.get("title") or record.get("file_name") or knowledge_id),
                description=str(record.get("description") or ""),
            )
            for record in payload["data"]
            if (knowledge_id := str(record.get("id", "")).strip())
        ]
        return SmartQKnowledgePage(items=items, **payload["pagination"])

    def get_chunk_page(
        self, knowledge_id: str | None, page: int = 1, page_size: int = DEFAULT_PAGE_SIZE
    ) -> SmartQChunkPage:
        """Retrieve a single page of processed text chunks from SmartQ."""

        knowledge_id = self._require_value(knowledge_id, "Knowledge ID")
        payload = self._get_page(
            f"/chunks/{quote(knowledge_id, safe='')}", page, page_size
        )
        items = [
            SmartQChunk(index=self._chunk_index(record, index), content=content)
            for index, record in enumerate(payload["data"])
            if (content := str(record.get("content") or "").strip())
        ]
        return SmartQChunkPage(items=items, **payload["pagination"])

    def get_article(self, knowledge_id: str | None) -> Article:
        """Build a Chinese Article with all text chunks for Q/A generation."""

        knowledge_id = self._require_value(knowledge_id, "Knowledge ID")
        encoded_id = quote(knowledge_id, safe="")
        knowledge = self._get_json(f"/knowledge/{encoded_id}").get("data")
        if not isinstance(knowledge, dict):
            raise SmartQAPIError("SmartQ returned an invalid knowledge response")

        raw_chunks = self._get_all_pages(
            f"/chunks/{encoded_id}", self._ARTICLE_CHUNK_PAGE_SIZE
        )
        raw_chunks.sort(key=lambda chunk: self._chunk_index(chunk, 0))
        chunks = [
            Chunk(
                heading=f"分块 {self._chunk_index(chunk, index) + 1}",
                level=1,
                content=content,
            )
            for index, chunk in enumerate(raw_chunks)
            if (content := str(chunk.get("content") or "").strip())
        ]
        if not chunks:
            raise SmartQAPIError(
                "This SmartQ knowledge has no processed text chunks. "
                "Wait for parsing to complete and try again."
            )

        title = str(knowledge.get("title") or knowledge.get("file_name") or knowledge_id)
        return Article(
            title=title,
            source=f"{self.api_url}/knowledge/{encoded_id}",
            language="cn",
            chunks=chunks,
            summary=str(knowledge.get("description") or ""),
        )

    @staticmethod
    def _require_value(value: str | None, label: str) -> str:
        if not value or not value.strip():
            raise SmartQAPIError(f"{label} cannot be empty")
        return value.strip()

    @staticmethod
    def _chunk_index(chunk: dict[str, Any], fallback: int) -> int:
        try:
            return int(chunk.get("chunk_index", fallback))
        except (TypeError, ValueError):
            return fallback

    def _get_page(
        self, path: str, page: int, page_size: int
    ) -> dict[str, Any]:
        if page < 1:
            raise SmartQAPIError("Page must be at least 1")
        if page_size < 1:
            raise SmartQAPIError("Page size must be at least 1")

        response = self._get_json(path, {"page": page, "page_size": page_size})
        data = response.get("data")
        if not isinstance(data, list):
            raise SmartQAPIError("SmartQ returned an invalid paginated response")

        total = response.get("total")
        if not isinstance(total, int):
            total = len(data)
        response_page = response.get("page")
        if not isinstance(response_page, int) or response_page < 1:
            response_page = page
        response_page_size = response.get("page_size")
        if not isinstance(response_page_size, int) or response_page_size < 1:
            response_page_size = page_size
        return {
            "data": [item for item in data if isinstance(item, dict)],
            "pagination": {
                "page": response_page,
                "page_size": response_page_size,
                "total": total,
            },
        }

    def _get_all_pages(self, path: str, page_size: int) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        page = 1
        while True:
            result = self._get_page(path, page, page_size)
            batch = result["data"]
            records.extend(batch)
            pagination = result["pagination"]
            if not batch or page * pagination["page_size"] >= pagination["total"]:
                return records
            page += 1

    def _get_json(
        self, path: str, query: dict[str, int] | None = None
    ) -> dict[str, Any]:
        url = f"{self.api_url}{path}"
        if query:
            url = f"{url}?{urlencode(query)}"
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "X-API-Key": self._api_key,
            },
        )
        try:
            with urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            raise SmartQAPIError(
                f"SmartQ API request failed with HTTP {error.code}"
            ) from error
        except URLError as error:
            raise SmartQAPIError(f"Unable to reach SmartQ API: {error.reason}") from error
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SmartQAPIError("SmartQ API returned an unreadable response") from error

        if not isinstance(payload, dict):
            raise SmartQAPIError("SmartQ API returned an invalid JSON response")
        if payload.get("success") is False:
            raise SmartQAPIError(str(payload.get("message") or "SmartQ API request failed"))
        return payload
