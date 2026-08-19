from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """Represent a section of an Article"""

    heading: str
    level: int
    content: str

    def __getitem__(self, key: str) -> str | int:
        return getattr(self, key)


class Article(BaseModel):
    """Wikipedia Article in WikiText format"""

    title: str
    source: str
    language: str
    chunks: list[Chunk]
    summary: str = ""

    def to_json(self) -> dict:
        return self.__dict__


class QA(BaseModel):
    """Question / Answer Pair"""

    type: str
    language: str
    article_title: str
    chunks: list[int]
    question: str
    answer: str

    def to_json(self) -> dict:
        return self.__dict__


class QAFormat(BaseModel):
    """
    A structured question-answer pair generated from a given context text.
    The question should be answerable directly from the context, and the answer
    must be a verbatim extract or a logical inference from the text.
    """

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
        description="The exact answer to the question, extracted or inferred from the context. "
        "Should be concise (1-2 sentences) and factually grounded in the text.",
        examples=[
            "he Eiffel Tower is named after the engineer Gustave Eiffel.",
            "The chemical energy is stored in carbohydrate molecules, such as sugars and starches.",
        ],
    )
