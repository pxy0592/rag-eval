import logging
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from lib.models.embedding import (
    EmbeddingModel,
    calculate_hybrid_scores,
    colbert_similarity,
    dense_similarity,
    sparse_similarity,
)
from lib.models.rerank import RerankerModel
from lib.types import Chunk, Document, RetrievedChunk

log = logging.getLogger("app")

class KnowledgeBase:
    def __init__(self, name: str, test: bool = False) -> None:
        assert name
        self.name = name
        self.db_path = (
            Path(__file__).resolve().parent.parent.parent
            / "data"
            / name
            / "vectordb.pkl"
        )
        self.test = test
        self.documents: list[Document] = []
        self.chunks: list[Chunk] = []
        self.dense_embeddings: NDArray | None = None
        self.sparse_embeddings: list[dict[str, float]] | None = None
        self.colbert_embeddings: list[NDArray] | None = None
        if not self.test:
            self.load()

    @property
    def is_empty(self) -> bool:
        return len(self.chunks) == 0

    def insert(
        self,
        docs: list[Document],
        chunks: list[Chunk],
        embedding_model: EmbeddingModel,
        batch_size: int = 32,
    ) -> None:
        assert len(docs) > 0 and len(chunks) > 0
        assert all(isinstance(d, Document) for d in docs)
        assert all(isinstance(c, Chunk) for c in chunks)
        assert embedding_model

        log.debug(f"Embedding {len(docs)} chunks.")

        start = time.time()
        result = embedding_model.encode(
            sentences=[chunk.text for chunk in chunks],
            return_dense=True,
            return_sparse=True,
            return_colbert=True,
            batch_size=batch_size,
        )
        log.debug(
            f"Finished embedding {len(docs)} chunks in {time.time() - start}"
        )

        dense, sparse, colbert = (
            result["dense"],
            result["sparse"],
            result["colbert"],
        )

        if len(self.documents) == 0:
            self.dense_embeddings = dense
            self.sparse_embeddings = sparse
            self.colbert_embeddings = colbert
        else:
            self.dense_embeddings = np.vstack([self.dense_embeddings, dense])
            self.sparse_embeddings.extend(sparse)
            # can't do vstack because colbert are unhomougenious
            self.colbert_embeddings.extend(colbert)

        self.documents.extend(docs)
        self.chunks.extend(chunks)

        if not self.test:
            self.save()

    def search(
        self,
        query: str,
        embedding_model: EmbeddingModel,
        reranker_model: RerankerModel | None,
        top_k: int = 20,
        top_r: int = 10,
        threshold: float = 0,
    ) -> list[RetrievedChunk]:
        """Hybrid Search with Reranking"""
        assert not self.is_empty
        assert 0 < threshold <= 1
        assert len(self.documents) > 0

        result = embedding_model.encode(
            [query],
            return_dense=True,
            return_sparse=True,
            return_colbert=True,
        )
        q_dense, q_sparse, q_colbert = (
            result["dense"],
            result["sparse"],
            result["colbert"],
        )

        dense_scores = dense_similarity(q_dense, self.dense_embeddings)[0]
        sparse_scores = sparse_similarity(q_sparse, self.sparse_embeddings)[0]
        colbert_scores = colbert_similarity(q_colbert, self.colbert_embeddings)[
            0
        ]
        hybrid_scores = calculate_hybrid_scores(
            scores=(dense_scores, sparse_scores, colbert_scores),
            weights=(0.4, 0.2, 0.4),
        )[0]
        rerank_scores = None
        scores, top_indices = torch.topk(
            hybrid_scores, k=min(top_k, len(hybrid_scores))
        )

        if reranker_model:
            top_texts = (self.chunks[idx].text for idx in top_indices.tolist())
            pairs = list((query, text) for text in top_texts)
            rerank_scores = reranker_model.compute_score(pairs)
            scores, top_indices = torch.topk(
                rerank_scores, k=min(top_r, len(rerank_scores))
            )

        retrieved_chunks = []
        for idx in top_indices[scores >= threshold].tolist():
            dense_score = dense_scores[idx].item()
            sparse_score = sparse_scores[idx].item()
            colbert_score = colbert_scores[idx].item()
            hybrid_score = hybrid_scores[idx].item()
            rerank_score = rerank_scores[idx].item() if reranker_model else None
            retrieved_chunks.append(
                RetrievedChunk(
                    chunk=self.chunks[idx],
                    scores={
                        "dense_score": dense_score,
                        "sparse_score": sparse_score,
                        "colbert_score": colbert_score,
                        "hybrid_score": hybrid_score,
                        "rerank_score": rerank_score,
                    },
                )
            )
        return retrieved_chunks

    def save(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "documents": self.documents,
            "chunks": self.chunks,
            "dense_embeddings": self.dense_embeddings,
            "sparse_embeddings": self.sparse_embeddings,
            "colbert_embeddings": self.colbert_embeddings,
        }
        with open(self.db_path, "wb") as f:
            pickle.dump(data, f)
        log.debug(f"Saved KnowledgeBase state to {self.db_path}")

    def load(self) -> None:
        if self.db_path.exists():
            with open(self.db_path, "rb") as f:
                data = pickle.load(f)
            self.documents = data.get("documents", [])
            self.chunks = data.get("chunks", [])
            self.dense_embeddings = data.get("dense_embeddings", [])
            self.sparse_embeddings = data.get("sparse_embeddings", [])
            self.colbert_embeddings = data.get("colbert_embeddings", [])
            log.debug(f"Loaded KnowledgeBase state from {self.db_path}")
        else:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            log.debug(f"No saved state found at {self.db_path}")
