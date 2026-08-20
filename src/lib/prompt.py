from typing import Final, TypedDict


class Prompt(TypedDict):
    factual_qa_pair: str
    factual_answer_rewrite: str


PROMPT: Final[Prompt] = {
    "factual_qa_pair": (
        "Generate one factual question and answer in the text's language.\n"
        "Use only information from this text.\n"
        "The question must be clear and self-contained.\n"
        "The answer must be a concise, standalone declarative sentence that "
        "restates the necessary subject, time, scope, and relationship from the "
        "question or source text. Do not return only a bare number, date, name, "
        "unit, or noun phrase. Preserve the source's exact factual value and unit.\n"
        "For example, for the Chinese question '截至2020年末，全段共有多少名现员？', "
        "answer '截至2020年末，锦州机务段共有4041名现员。', not only '4041人'.\n"
        "Keep the answer focused, normally one sentence and at most two sentences.\n"
        "Return only a valid JSON object with exactly these string keys: "
        '{{"question":"...","answer":"..."}}. '
        "Do not use Markdown, code fences, or labels such as Question/Answer or 问题/答案.\n\n"
        "Source title: {article_title}\n"
        "Text:\n{context}\n"
    ),
    "factual_answer_rewrite": (
        "Rewrite the current factual answer in the source text's language.\n"
        "Keep the question unchanged and use only facts supported by the source text.\n"
        "Produce a concise, standalone declarative sentence that explicitly states "
        "the necessary subject, time, scope, and relationship. Do not return only a "
        "bare number, date, name, unit, or noun phrase. Preserve the exact value and unit.\n"
        "For example, rewrite '4041人' as "
        "'截至2020年末，锦州机务段共有4041名现员。' when supported by the context.\n"
        "Return only a valid JSON object with exactly these string keys: "
        '{{"question":"...","answer":"..."}}. '
        "Do not use Markdown, code fences, or labels.\n\n"
        "Source title: {article_title}\n"
        "Question: {question}\n"
        "Current answer: {answer}\n"
        "Text:\n{context}\n"
    ),
}
