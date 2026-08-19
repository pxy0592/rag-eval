from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from lib.types import Chunk, Document

# Types ------------------------------------------------------------------


# TODO: combine the rag-eval representations with the local representations
# TODO: change ids to be int by default
class DatasetChunk(BaseModel):
    heading: str
    level: int
    content: str


class DatasetDocument(BaseModel):
    title: str
    source: str
    language: str
    chunks: list[DatasetChunk]

    @staticmethod
    def load(path: Path) -> tuple[list[Document], list[Chunk]]:
        df = pd.read_json(path, lines=True)
        docs = [
            DatasetDocument.model_validate(row.to_dict())
            for _, row in df.iterrows()
        ]

        documents, chunks = [], []
        for id, doc in enumerate(docs):
            documents.append(
                Document(
                    id=str(id),
                    title=doc.title,
                    language=doc.language,
                    source=doc.source,
                )
            )
            chunks.extend([
                Chunk(
                    id=str(chunk_id),
                    doc_id=str(id),
                    page=-1,
                    text=chunk.content,
                )
                for chunk_id, chunk in enumerate(doc.chunks)
            ])

        return documents, chunks


class DatasetQA(BaseModel):
    type: str
    language: str
    article_title: str
    chunks: list[int]
    question: str
    answer: str

    @staticmethod
    def load(path: Path) -> list["DatasetQA"]:
        df = pd.read_json(path, lines=True)
        return [
            DatasetQA.model_validate(row.to_dict()) for _, row in df.iterrows()
        ]


class EvaluationDataset(BaseModel):
    user_input: str
    retrieved_contexts: list[str]
    retrieved_ids: list[int]
    ground_truth_ids: list[int]
    response: str
    reference: str

    @staticmethod
    def load(path: Path) -> list["EvaluationDataset"]:
        df = pd.read_json(path, lines=False)
        return [
            EvaluationDataset.model_validate(row.to_dict())
            for _, row in df.iterrows()
        ]
