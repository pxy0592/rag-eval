import re

import requests
import wikipediaapi as wiki

from .types import Article, Chunk
from .utils import clean_text, split_text


class WikipediaProcessor:
    """Fetch and parse Wikipedia articles through the MediaWiki API."""

    api_url = "https://{lang}.wikipedia.org/w/api.php"

    def _request(self, lang: str, params: dict) -> dict:
        response = requests.get(self.api_url.format(lang=lang), params=params)
        response.raise_for_status()
        return response.json()

    def _get_page_id(self, title: str, lang: str) -> int | None:
        data = self._request(lang, {
            "action": "query",
            "list": "search",
            "srsearch": title,
            "format": "json",
        })
        search_results = data.get("query", {}).get("search", [])
        return search_results[0].get("pageid") if search_results else None

    def _get_lang_link(self, page_id: int, lang: str) -> str | None:
        data = self._request("en", {
            "action": "query",
            "pageids": page_id,
            "prop": "langlinks",
            "lllang": lang,
            "format": "json",
        })
        pages = data.get("query", {}).get("pages", {})
        page = next(iter(pages.values()), {})
        links = page.get("langlinks", [])
        return links[0].get("*") if links else None

    def _get_page_content(self, page_id: int, lang: str) -> str:
        data = self._request(lang, {
            "action": "query",
            "pageids": page_id,
            "prop": "revisions",
            "rvprop": "content",
            "rvslots": "main",
            "format": "json",
        })
        pages = data.get("query", {}).get("pages", {})
        page = next(iter(pages.values()), {})
        revisions = page.get("revisions", [])
        return revisions[0]["slots"]["main"]["*"] if revisions else ""

    def _get_page_title(self, page_id: int, lang: str) -> str:
        data = self._request(lang, {
            "action": "query",
            "pageids": page_id,
            "format": "json",
        })
        pages = data.get("query", {}).get("pages", {})
        return next(iter(pages.values()), {}).get("title", "")

    def _parse_sections(self, wikitext: str) -> list[Chunk]:
        sections: list[Chunk] = []
        matches = list(re.finditer(r"^[ \t]*(={2,6})\s*(.+?)\s*\1\s*$", wikitext, re.MULTILINE))
        for index, match in enumerate(matches):
            content_start = match.end()
            content_end = matches[index + 1].start() if index + 1 < len(matches) else len(wikitext)
            content = clean_text(wikitext[content_start:content_end])
            if content:
                sections.append(Chunk(
                    heading=match.group(2).strip(),
                    level=len(match.group(1)) - 1,
                    content=content,
                ))
        return sections

    def _split_text(self, text: str, max_size: int) -> list[str]:
        return split_text(text, max_size)

    def get_article(self, title: str, lang: str = "en", max_chunk_size: int = 2000) -> Article | None:
        page_id = self._get_page_id(title, lang)
        if page_id is None:
            return None
        if lang != "en":
            translated_title = self._get_lang_link(page_id, lang)
            if not translated_title:
                return None
            page_id = self._get_page_id(translated_title, lang)
            if page_id is None:
                return None
        content = self._get_page_content(page_id, lang)
        chunks = self._parse_sections(content)
        resized_chunks = [
            Chunk(heading=chunk.heading, level=chunk.level, content=part)
            for chunk in chunks
            for part in self._split_text(chunk.content, max_chunk_size)
        ]
        return Article(
            title=self._get_page_title(page_id, lang),
            source=f"https://{lang}.wikipedia.org/?curid={page_id}",
            language=lang,
            chunks=resized_chunks,
        )

wk = wiki.Wikipedia(user_agent="SmartQ Dataset Generator (merlin@example.com)", language="en")


def get_wikipedia_article(
    title: str, langs: str | list[str] = "en", max_chunk_size: int = 2000
) -> list[Article]:
    """Fetch and chunk a Wikipedia article"""
    langs = [langs] if isinstance(langs, str) else langs

    # Theres only needed to fetch the en version
    page = wk.page(title)
    pages = [page] if page.exists() else []

    for lang in filter(lambda x: x != "en", langs):
        lang_page = page.langlinks.get(lang)
        if lang_page:
            pages.append(lang_page)

    return [
        Article(
            title=page.title,
            source=page.fullurl or "unknow",
            language=page.language,
            chunks=_get_chunks(page.sections, max_chunk_size),
            summary=clean_text(page.summary),
        )
        for page in pages
    ]


def _get_chunks(
    sections: list[wiki.WikipediaPageSection],
    max_chunk_size: int = 300,
    level: int = 0,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in sections:
        cleaned_text = clean_text(section.text)
        if len(cleaned_text) == 0:
            continue
        if len(cleaned_text) > max_chunk_size:
            chunk_parts = split_text(cleaned_text, max_chunk_size)
            for part in chunk_parts:
                chunks.append({
                    "heading": section.title,
                    "level": level + 1,
                    "content": part,
                })
        else:
            chunks.append({
                "heading": section.title,
                "level": level + 1,
                "content": cleaned_text,
            })
        subsections = _get_chunks(section.sections, level + 1)
        chunks.extend(subsections)
    return chunks
