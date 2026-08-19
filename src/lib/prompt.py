from typing import Final, TypedDict


class Prompt(TypedDict):
    factual_qa_pair: str


PROMPT: Final[Prompt] = {
    "factual_qa_pair": (
        "Generate one factual question and answer in the text's language.\n"
        "Use only information from this text.\n"
        "Return only a valid JSON object with exactly these string keys: "
        '{{"question":"...","answer":"..."}}. '
        "Do not use Markdown, code fences, or labels such as Question/Answer or 问题/答案.\n\n"
        "Text:\n{context}\n"
    )
}
