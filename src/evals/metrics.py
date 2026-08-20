"""Deterministic retrieval and generation metrics for evaluation runs."""

from __future__ import annotations

from collections import Counter
import unicodedata
from collections.abc import Callable, Iterable

import numpy as np


DEFAULT_RETRIEVAL_CUTOFFS = (1, 5, 10)
CHUNK_INDEX_TOLERANCE = 5
GENERATION_METRICS = ("answer_exact_match", "answer_character_f1")


def _tolerant_relevance(
    predictions: list[int],
    truths: list[int],
    cutoff: int,
    tolerance: int = CHUNK_INDEX_TOLERANCE,
) -> tuple[list[bool], int, int]:
    """Match ranked predictions to unique truths once within an inclusive range."""
    unique_truths = list(dict.fromkeys(truths))
    unmatched = set(range(len(unique_truths)))
    relevance: list[bool] = []

    for prediction in predictions[:cutoff]:
        candidates = [
            (abs(prediction - unique_truths[index]), index)
            for index in unmatched
            if abs(prediction - unique_truths[index]) <= tolerance
        ]
        if not candidates:
            relevance.append(False)
            continue
        _, matched_index = min(candidates)
        unmatched.remove(matched_index)
        relevance.append(True)

    return relevance, len(unique_truths) - len(unmatched), len(unique_truths)


def calc_precision(
    preds: list[list[int]], truths: list[list[int]], cutoffs: list[int]
) -> np.ndarray:
    """Calculate precision@k using one-to-one chunk matches within ±5 indexes."""
    precision = np.zeros(len(cutoffs))
    for pred, truth in zip(preds, truths, strict=True):
        for index, cutoff in enumerate(cutoffs):
            relevance, _, _ = _tolerant_relevance(pred, truth, cutoff)
            precision[index] += sum(relevance) / cutoff
    return precision / len(preds) if preds else precision


def calc_recall(
    preds: list[list[int]], truths: list[list[int]], cutoffs: list[int]
) -> np.ndarray:
    """Calculate recall@k using unique truth coverage within ±5 indexes."""
    recall = np.zeros(len(cutoffs))
    for pred, truth in zip(preds, truths, strict=True):
        for index, cutoff in enumerate(cutoffs):
            _, matched_truths, truth_count = _tolerant_relevance(
                pred, truth, cutoff
            )
            recall[index] += matched_truths / max(truth_count, 1)
    return recall / len(preds) if preds else recall


def calc_mrr(
    preds: list[list[int]], truths: list[list[int]], cutoffs: list[int]
) -> np.ndarray:
    """Calculate MRR@k with the first one-to-one match within ±5 indexes."""
    mrr = np.zeros(len(cutoffs))
    for pred, truth in zip(preds, truths, strict=True):
        for index, cutoff in enumerate(cutoffs):
            relevance, _, _ = _tolerant_relevance(pred, truth, cutoff)
            reciprocal_rank = next(
                (1.0 / rank for rank, relevant in enumerate(relevance, 1) if relevant),
                0.0,
            )
            mrr[index] += reciprocal_rank
    return mrr / len(preds) if preds else mrr


def calc_ndcg(
    preds: list[list[int]], truths: list[list[int]], cutoffs: list[int]
) -> np.ndarray:
    """Calculate binary NDCG@k with one-to-one matches within ±5 indexes."""
    ndcg = np.zeros(len(cutoffs))
    for pred, truth in zip(preds, truths, strict=True):
        for index, cutoff in enumerate(cutoffs):
            relevance, _, truth_count = _tolerant_relevance(pred, truth, cutoff)
            dcg = sum(
                1.0 / np.log2(rank + 1)
                for rank, relevant in enumerate(relevance, start=1)
                if relevant
            )
            ideal_hits = min(truth_count, cutoff)
            idcg = sum(
                1.0 / np.log2(rank + 1)
                for rank in range(1, ideal_hits + 1)
            )
            ndcg[index] += dcg / idcg if idcg else 0.0
    return ndcg / len(preds) if preds else ndcg


def calc_map(
    preds: list[list[int]], truths: list[list[int]], cutoffs: list[int]
) -> np.ndarray:
    """Calculate MAP@k with one-to-one chunk matches within ±5 indexes."""
    mean_average_precision = np.zeros(len(cutoffs))
    for pred, truth in zip(preds, truths, strict=True):
        for index, cutoff in enumerate(cutoffs):
            relevance, _, truth_count = _tolerant_relevance(pred, truth, cutoff)
            hits = 0
            precision_sum = 0.0
            for rank, relevant in enumerate(relevance, start=1):
                if relevant:
                    hits += 1
                    precision_sum += hits / rank
            denominator = min(truth_count, cutoff)
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
