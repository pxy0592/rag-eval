from src.lib.prompt import PROMPT


def test_factual_prompt_requires_contextual_sentence_answer():
    prompt = PROMPT["factual_qa_pair"].format(
        article_title="锦州机务段",
        context="截至2020年末，全段现员4041人。",
    )

    assert "Do not return only a bare number" in prompt
    assert "截至2020年末，锦州机务段共有4041名现员。" in prompt
    assert "Source title: 锦州机务段" in prompt


def test_rewrite_prompt_preserves_question_and_uses_current_answer():
    prompt = PROMPT["factual_answer_rewrite"].format(
        article_title="锦州机务段",
        question="截至2020年末，全段共有多少名现员？",
        answer="4041人",
        context="截至2020年末，全段现员4041人。",
    )

    assert "Keep the question unchanged" in prompt
    assert "Current answer: 4041人" in prompt
    assert "standalone declarative sentence" in prompt
