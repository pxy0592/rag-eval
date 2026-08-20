"""Deterministic retrieval and generation metrics for evaluation runs."""

from __future__ import annotations

from collections import Counter
import unicodedata
from collections.abc import Callable, Iterable

import numpy as np


DEFAULT_RETRIEVAL_CUTOFFS = (1, 5, 10)
GENERATION_METRICS = ("answer_exact_match", "answer_character_f1")


def calc_precision(
    preds: list[list[int]], truths: list[list[int]], cutoffs: list[int]
) -> np.ndarray:
    """Calculate precision@k for every requested cutoff."""
    precision = np.zeros(len(cutoffs))
    for pred, truth in zip(preds, truths, strict=True):
        truth_set = set(truth)
        for index, cutoff in enumerate(cutoffs):
            precision[index] += sum(
                item in truth_set for item in pred[:cutoff]
            ) / cutoff
    return precision / len(preds) if preds else precision


def calc_recall(
    preds: list[list[int]], truths: list[list[int]], cutoffs: list[int]
) -> np.ndarray:
    """Calculate recall@k for every requested cutoff."""
    recall = np.zeros(len(cutoffs))
    for pred, truth in zip(preds, truths, strict=True):
        truth_set = set(truth)
        for index, cutoff in enumerate(cutoffs):
            recall[index] += sum(
                item in truth_set for item in pred[:cutoff]
            ) / max(len(truth_set), 1)
    return recall / len(preds) if preds else recall


def calc_mrr(
    preds: list[list[int]], truths: list[list[int]], cutoffs: list[int]
) -> np.ndarray:
    """Calculate mean reciprocal rank@k for every requested cutoff."""
    mrr = np.zeros(len(cutoffs))
    for pred, truth in zip(preds, truths, strict=True):
        truth_set = set(truth)
        for index, cutoff in enumerate(cutoffs):
            reciprocal_rank = 0.0
            for rank, item in enumerate(pred[:cutoff], start=1):
                if item in truth_set:
                    reciprocal_rank = 1.0 / rank
                    break
            mrr[index] += reciprocal_rank
    return mrr / len(preds) if preds else mrr


def calc_ndcg(
    preds: list[list[int]], truths: list[list[int]], cutoffs: list[int]
) -> np.ndarray:
    """Calculate binary-relevance normalized DCG@k."""
    ndcg = np.zeros(len(cutoffs))
    for pred, truth in zip(preds, truths, strict=True):
        truth_set = set(truth)
        for index, cutoff in enumerate(cutoffs):
            dcg = sum(
                1.0 / np.log2(rank + 1)
                for rank, item in enumerate(pred[:cutoff], start=1)
                if item in truth_set
            )
            ideal_hits = min(len(truth_set), cutoff)
            idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
            ndcg[index] += dcg / idcg if idcg else 0.0
    return ndcg / len(preds) if preds else ndcg


def calc_map(
    preds: list[list[int]], truths: list[list[int]], cutoffs: list[int]
) -> np.ndarray:
    """Calculate mean average precision@k."""
    mean_average_precision = np.zeros(len(cutoffs))
    for pred, truth in zip(preds, truths, strict=True):
        truth_set = set(truth)
        for index, cutoff in enumerate(cutoffs):
            hits = 0
            precision_sum = 0.0
            for rank, item in enumerate(pred[:cutoff], start=1):
                if item in truth_set:
                    hits += 1
                    precision_sum += hits / rank
            denominator = min(len(truth_set), cutoff)
            mean_average_precision[index] += (
                precision_sum / denominator if denominator else 0.0
            )
    return (
        mean_average_precision / len(preds)
        if preds
        else mean_average_precision
    )


def normalize_answer(value: str) -> str:
    """Normalize answer text for deterministic cross-language comparisons."""
    return "".join(
        char.casefold()
        for char in unicodedata.normalize("NFKC", value)
        if not char.isspace() and not unicodedata.category(char).startswith("P")
    )


def answer_exact_match(prediction: str, reference: str) -> float:
    """Return 1 when normalized answers match exactly, else 0."""
    normalized_reference = normalize_answer(reference)
    return float(bool(normalized_reference) and normalize_answer(prediction) == normalized_reference)


def answer_character_f1(prediction: str, reference: str) -> float:
    """Return multiset character F1 after deterministic normalization."""
    predicted = normalize_answer(prediction)
    expected = normalize_answer(reference)
    if not predicted or not expected:
        return 0.0
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def supported_metric_names(
    cutoffs: Iterable[int] = DEFAULT_RETRIEVAL_CUTOFFS,
) -> list[str]:
    """Return all deterministic metric names understood by the scoring CLI."""
    retrieval = [
        f"{name}@{cutoff}"
        for name in ("precision", "recall", "mrr", "ndcg", "map")
        for cutoff in cutoffs
    ]
    return retrieval + list(GENERATION_METRICS)


def select_metrics(raw_metrics: str | Iterable[str]) -> list[str]:
    """Validate an `all` value or a comma-separated/iterable metric selection."""
    supported = supported_metric_names()
    if isinstance(raw_metrics, str):
        values = [value.strip() for value in raw_metrics.split(",") if value.strip()]
    else:
        values = [value.strip() for value in raw_metrics if value.strip()]
    if values == ["all"]:
        return supported
    if not values:
        raise ValueError("at least one metric or 'all' must be selected")
    unknown = sorted(set(values) - set(supported))
    if unknown:
        raise ValueError(
            "unsupported metrics: "
            + ", ".join(unknown)
            + "; supported metrics: "
            + ", ".join(supported)
        )
    return list(dict.fromkeys(values))


RETRIEVAL_METRICS: dict[str, Callable[[list[list[int]], list[list[int]], list[int]], np.ndarray]] = {
    "precision": calc_precision,
    "recall": calc_recall,
    "mrr": calc_mrr,
    "ndcg": calc_ndcg,
    "map": calc_map,
}
