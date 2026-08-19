import numpy as np


def calc_precision(preds: list[list[int]], truths: list[list[int]], cutoffs: list[int]) -> np.ndarray:
    """Precision@k: Ratio of relevant documents among the retrived documents

    Args:
        preds (list[list[int]]): A list of predicted rankings per user.
        truths (list[list[int]]): A list of ground truth relevant items per user.
        cutoffs (list[int]): A list of cutoff values (e.g. [1, 5, 10]).

    Returns:
        np.ndarray: Precision at each cutoff, shape (len(cutoffs),).
    """
    precision = np.zeros(len(cutoffs))

    for pred, truth in zip(preds, truths):
        truth_set = set(truth)
        for i, k in enumerate(cutoffs):
            top_k = pred[:k]
            correct = sum(1 for item in top_k if item in truth_set)
            precision[i] += correct / k

    precision /= len(preds)
    return precision


def calc_recall(preds: list[list[int]], truths: list[list[int]], cutoffs: list[int]) -> np.ndarray:
    """Recall@k: Ratio of relevant documents among the total documents

    Args:
        preds (list[list[int]]): A list of predicted rankings per user.
        truths (list[list[int]]): A list of ground truth relevant items per user.
        cutoffs (list[int]): A list of cutoff values (e.g. [1, 5, 10]).

    Returns:
        np.ndarray: Recall at each cutoff, shape (len(cutoffs),).
    """
    recalls = np.zeros(len(cutoffs))

    for pred, truth in zip(preds, truths):
        truth_set = set(truth)
        for i, k in enumerate(cutoffs):
            top_k = pred[:k]
            correct = sum(1 for item in top_k if item in truth_set)
            recalls[i] += correct / max(len(truth_set), 1)

    recalls /= len(preds)
    return recalls


def calc_mrr(preds: list[list[int]], truths: list[list[int]], cutoffs: list[int]):
    """Mean Reciprocal Rank: Measures the rank position of the first relevant result

    Args:
        preds (list[list[int]]): A list of predicted rankings per user.
        truths (list[list[int]]): A list of ground truth relevant items per user
        cutoffs (list[int]): A list of cutoff values (e.g. [1, 5, 10]).

    Returns:
        np.ndarray: MRR at each cutoff, shape (len(cutoffs),).
    """
    MRRs = np.zeros(len(cutoffs))

    for pred, truth in zip(preds, truths):
        truth_set = set(truth)
        for i, k in enumerate(cutoffs):
            top_k = pred[:k]
            reciprocal_rank = 0.0
            for rank, item in enumerate(top_k, start=1):
                if item in truth_set:
                    reciprocal_rank = 1.0 / rank
                    break
            MRRs[i] += reciprocal_rank

    MRRs /= len(preds)
    return MRRs


def calc_ndcg(preds: list[list[int]], truths: list[list[int]], cutoffs: list[int]):
    """Normalized Discounted Cumulative Gain: Ranking quality of all retrieved documents considering both order and relevance

    Args:
        preds (list[list[int]]): A list of predicted rankings per user.
        truths (list[list[int]]): A list of ground truth relevant items per user.
        cutoffs (list[int]): A list of cutoff values (e.g. [1, 5, 10]).

    Returns:
        np.ndarray: NDCG at each cutoff, shape (len(cutoffs),).
    """
    ndcgs = np.zeros(len(cutoffs))

    for pred, truth in zip(preds, truths):
        truth_set = set(truth)
        for i, k in enumerate(cutoffs):
            top_k = pred[:k]

            # DCG
            dcg = 0.0
            for rank, item in enumerate(top_k, start=1):
                if item in truth_set:
                    dcg += 1.0 / np.log2(rank + 1)

            # Ideal DCG (IDCG)
            ideal_hits = min(len(truth_set), k)
            idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))

            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcgs[i] += ndcg

    ndcgs /= len(preds)
    return ndcgs


def calc_map(preds: list[list[int]], truths: list[list[int]], cutoffs: list[int]) -> np.ndarray:
    """Mean Average Precision

    Args:
        preds (list[list[int]]): A list of predicted rankings per user.
        truths (list[list[int]]): A list of ground truth relevant items per user.
        cutoffs (list[int]): A list of cutoff values (e.g. [1, 5, 10]).

    Returns:
        np.ndarray: MAP at each cutoff, shape (len(cutoffs),).
    """
    maps = np.zeros(len(cutoffs))

    for pred, truth in zip(preds, truths):
        truth_set = set(truth)
        if not truth_set:
            continue  # skip to avoid division by zero

        for i, k in enumerate(cutoffs):
            top_k = pred[:k]
            num_hits = 0
            ap_sum = 0.0

            for idx, item in enumerate(top_k):
                if item in truth_set:
                    num_hits += 1
                    precision_at_i = num_hits / (idx + 1)
                    ap_sum += precision_at_i

            map_k = ap_sum / min(len(truth_set), k) if num_hits > 0 else 0.0
            maps[i] += map_k

    maps /= len(preds)
    return maps
