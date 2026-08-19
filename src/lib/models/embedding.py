from typing import TypedDict

import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel
from numpy.typing import NDArray
from torch.nn.utils.rnn import pad_sequence

from lib.settings import settings


class EmbeddingOutput(TypedDict):
    dense: NDArray
    sparse: list[dict[str, float]]
    colbert: list[NDArray[np.float16]]


class EmbeddingModel:
    def __init__(self) -> None:
        self._model = BGEM3FlagModel(
            model_name_or_path=settings.EMBEDDING_MODEL,
            normalize_embeddings=True,
            use_fp16=True,
            devices=settings.DEVICE,
        )

    def encode(
        self,
        sentences: list[str],
        return_dense: bool = True,
        return_sparse: bool = False,
        return_colbert: bool = False,
        batch_size: int = 32,
        **kwargs: dict,
    ) -> EmbeddingOutput:
        assert isinstance(sentences, list)
        assert len(sentences) > 0
        assert return_dense or return_sparse or return_colbert

        result = self._model.encode(
            sentences,
            batch_size=batch_size,
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert_vecs=return_colbert,
            max_length=settings.EMBEDDING_TOKEN_LIMIT,
            convert_to_numpy=False,
            **kwargs,
        )
        return {
            "dense": result.get("dense_vecs", None),
            "sparse": result.get("lexical_weights", None),
            "colbert": result.get("colbert_vecs", None),
        }  # type: ignore


def dense_similarity(
    embeddings_1: NDArray,
    embeddings_2: NDArray,
) -> torch.Tensor:
    assert isinstance(embeddings_1, np.ndarray) and isinstance(
        embeddings_2, np.ndarray
    )

    # Convert to float16 tensors
    e1 = torch.from_numpy(embeddings_1).to(torch.float16)
    e2 = torch.from_numpy(embeddings_2).to(torch.float16)

    e1 = torch.nn.functional.normalize(e1, p=2, dim=1)
    e2 = torch.nn.functional.normalize(e2, p=2, dim=1)

    # dot product (embeddings are normalized)
    scores = e1 @ e2.T
    return scores


# BUG: evaluate a better way to compute this similarity
def sparse_similarity(
    lexical_weights_1: list[dict[str, float]],
    lexical_weights_2: list[dict[str, float]],
) -> torch.Tensor:
    assert len(lexical_weights_1) > 0 and len(lexical_weights_2) > 0

    def compute_single(lw1: dict[str, float], lw2: dict[str, float]) -> float:
        dot_product = sum(
            weight * lw2.get(token, 0) for token, weight in lw1.items()
        )
        # cosine normalize by L2 norms of sparse vectors
        norm1 = np.sqrt(sum(w**2 for w in lw1.values()))
        norm2 = np.sqrt(sum(w**2 for w in lw2.values()))
        return dot_product / (norm1 * norm2 + 1e-8)

    scores_array = [
        [compute_single(lw1, lw2) for lw2 in lexical_weights_2]
        for lw1 in lexical_weights_1
    ]
    scores = torch.tensor(scores_array, dtype=torch.float16)
    return scores


def colbert_similarity(
    query_embs: list[NDArray],
    passage_embs: list[NDArray],
) -> torch.Tensor:
    assert  isinstance(query_embs, list)
    assert  isinstance(passage_embs, list)

    # Convert to tensors and get lengths
    queries = [torch.from_numpy(e).float() for e in query_embs]
    passages = [torch.from_numpy(e).float() for e in passage_embs]

    q_lens = torch.tensor([q.shape[0] for q in queries], dtype=torch.float16)
    p_lens = torch.tensor([p.shape[0] for p in passages], dtype=torch.float16)

    # Pad sequences
    q_padded = pad_sequence(queries, batch_first=True)  # (n_q, max_q_len, dim)
    p_padded = pad_sequence(passages, batch_first=True)  # (n_p, max_p_len, dim)

    # Compute similarity matrix
    # (n_q, max_q_len, dim) @ (n_p, dim, max_p_len) -> (n_q, n_p, max_q_len, max_p_len)
    scores = torch.einsum("qid,pjd->qpij", q_padded, p_padded)

    # Create masks
    q_mask = (
        torch.arange(q_padded.shape[1])[None, :] < q_lens[:, None]
    )  # (n_q, max_q_len)
    p_mask = (
        torch.arange(p_padded.shape[1])[None, :] < p_lens[:, None]
    )  # (n_p, max_p_len)

    # Apply masks
    scores = scores.masked_fill(
        ~q_mask[:, None, :, None] | ~p_mask[None, :, None, :], -torch.inf
    )

    # MaxSim over passage tokens, sum over query tokens
    max_scores = scores.max(dim=-1).values  # (n_q, n_p, max_q_len)
    summed_scores = max_scores.sum(dim=-1)  # (n_q, n_p)

    # Normalize by query lengths
    return (summed_scores / q_lens[:, None].float()).to(torch.float16)


def calculate_hybrid_scores(
    scores: tuple[torch.Tensor, ...],
    weights: tuple[float, ...],
) -> torch.Tensor:
    """Calculate hybrid retrieval scores by combining multiple similarity scores with weights."""
    assert len(scores) == len(weights)
    assert len(scores) > 0
    assert sum(weights) == 1
    assert all(score.dtype == torch.float16 for score in scores)

    stacked_scores = torch.stack(scores)
    weights_tensor = torch.tensor(weights, dtype=torch.float16).view(-1, 1, 1)
    hybrid_scores = (stacked_scores * weights_tensor).sum(dim=0)
    return hybrid_scores
