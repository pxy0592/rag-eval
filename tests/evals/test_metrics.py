import pytest

from src.evals.metrics import (
    answer_character_f1,
    answer_exact_match,
    calc_map,
    calc_mrr,
    calc_ndcg,
    calc_precision,
    calc_recall,
    select_metrics,
)


def test_retrieval_metrics_calculate_expected_scores_at_cutoff():
    predictions = [[1, 9], [9, 2]]
    expected = [[1], [2]]

    assert calc_precision(predictions, expected, [1]).tolist() == [0.5]
    assert calc_recall(predictions, expected, [1]).tolist() == [0.5]
    assert calc_mrr(predictions, expected, [2]).tolist() == [0.75]
    assert calc_ndcg(predictions, expected, [2]).tolist() == pytest.approx([0.8154648768])
    assert calc_map(predictions, expected, [2]).tolist() == pytest.approx([0.75])


def test_generation_metrics_normalize_chinese_punctuation_and_partial_overlap():
    assert answer_exact_match("4041 人。", "4041人") == 1.0
    assert answer_character_f1("4041人", "4041人") == 1.0
    assert answer_character_f1("4041", "4041人") == pytest.approx(8 / 9)


def test_metric_selection_supports_all_and_rejects_unknown_metric():
    assert "precision@1" in select_metrics("all")
    assert select_metrics("precision@1,answer_exact_match") == [
        "precision@1",
        "answer_exact_match",
    ]
    with pytest.raises(ValueError, match="unsupported metrics"):
        select_metrics("not-a-metric")
