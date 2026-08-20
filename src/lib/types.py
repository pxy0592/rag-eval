from dataclasses import dataclass
from time import time
from typing import Literal, NotRequired, TypedDict

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A chunk from either a source article or a retrievable document."""

    heading: str | None = None
    level: int | None = None
    content: str | None = None
    id: str | None = None
    doc_id: str | None = None
    page: int | None = None
    text: str | None = None

    def __getitem__(self, key: str) -> str | int | None:
        return getattr(self, key)

    def model_dump(self, **kwargs):
        """Omit fields that are irrelevant to the chunk representation in use."""
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(**kwargs)


class Metadata(TypedDict):
    created_at: str


class Document(BaseModel):
    id: str
    source: str
    title: str = "unknown"
    language: str = "en"
    metadata: Metadata = Field(
        default_factory=lambda: {"created_at": str(time())}
    )


class Scores(TypedDict):
    dense_score: float
    sparse_score: float
    colbert_score: float
    hybrid_score: float
    rerank_score: float | None


@dataclass
class RetrievedChunk:
    chunk: Chunk
    scores: Scores

    def __hash__(self) -> int:
        return hash(self.chunk.id)

    def __repr__(self) -> str:
        rerank_score = self.scores["rerank_score"]
        rerank = "n/a" if rerank_score is None else f"{rerank_score:.3f}"
        return (
            f"Source   {self.chunk.doc_id}\n"
            f"Dense:   {self.scores['dense_score']:.3f}, "
            f"Sparse:  {self.scores['sparse_score']:.3f}, "
            f"Colbert: {self.scores['colbert_score']:.3f}\n"
            f"Hybrid:  {self.scores['hybrid_score']:.3f}\n"
            f"Rerank:  {rerank}\n"
            f"Text:\n  {(self.chunk.text or '')[:300]} ..."
        )


class Message(TypedDict):
    text: str
    files: NotRequired[list[str]]


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str | tuple[str, ...]


class GenerationParams(TypedDict, total=False):
    max_tokens: int
    temperature: float
    top_p: float
    frequency_penalty: float
    presence_penalty: float
    repetition_penalty: float


class Article(BaseModel):
    """Wikipedia Article in WikiText format."""

    title: str
    source: str
    language: str
    chunks: list[Chunk]
    summary: str = ""

    def to_json(self) -> dict:
        return self.model_dump()


class QA(BaseModel):
    """Question / Answer Pair."""

    type: str
    language: str
    article_title: str
    chunks: list[int]
    question: str
    answer: str

    def to_json(self) -> dict:
        return self.model_dump()


class QAFormat(BaseModel):
    """A structured question-answer pair generated from a context text."""

    question: str = Field(
        ...,
        description="A clear, self-contained question derived from the context. "
        "Should be answerable with 1-2 sentences from the text. "
        "Avoid yes/no questions unless explicitly supported by context.",
        examples=[
            "Who is the Eiffel Tower named after?",
            "What type of molecules store the chemical energy produced during photosynthesis?",
        ],
    )
    answer: str = Field(
        ...,
        description="A concise, standalone declarative answer grounded in the text. "
        "It should restate the necessary subject, time, scope, and relationship "
        "instead of returning only a bare value, date, name, unit, or noun phrase. "
        "Preserve exact factual values and use 1-2 sentences.",
        examples=[
            "The Eiffel Tower is named after the engineer Gustave Eiffel.",
            "截至2020年末，锦州机务段共有4041名现员。",
        ],
    )
