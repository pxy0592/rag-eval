import json
import re
import tempfile
from typing import Any

from .types import Chunk


def format_context(context: list[Chunk]) -> str:
    context_str = ""
    for chunk in context:
        content = chunk["content"]  # clean it
        context_str += f"Heading: {chunk['heading']} \n\n {content} \n"
    return context_str


def create_json_file(data: Any, prefix: str = "data_") -> str:
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".json", prefix=prefix, delete=False, encoding="utf-8"
    ) as temp_f:
        json.dump(data, temp_f, indent=2, ensure_ascii=False)
        return temp_f.name


def parse_qa_output(llm_output: str | None) -> dict[str, str] | None:
    """
    Parses the LLM output string to extract the Question and Answer.

    Args:
        llm_output: The raw string output from the Language Model.

    Returns:
        A dictionary with 'question' and 'answer' keys if parsing is successful,
        otherwise None.
    """
    # --- Regex Explanation ---
    # r"..." : Raw string to avoid issues with backslashes
    # Question: : Matches the literal "Question:" label.
    # \s* : Matches zero or more whitespace characters (spaces, tabs, newlines).
    #       Handles potential spaces/newlines after the label.
    # (.*?) : Capturing Group 1 (The Question)
    #   . : Matches any character (except newline by default)
    #   * : Matches the previous character zero or more times
    #   ? : Makes the '*' non-greedy, so it stops matching as soon
    #       as the next part of the pattern (Answer:) is found.
    # \s* : Matches whitespace between the end of the question and the Answer label.
    # Answer: : Matches the literal "Answer:" label.
    # \s* : Matches whitespace after the Answer label.
    # (.*) : Capturing Group 2 (The Answer)
    #   . : Matches any character (including newline because of re.DOTALL)
    #   * : Matches zero or more times (greedy - takes rest of the string)
    # re.IGNORECASE : Makes "Question:" and "Answer:" matching case-insensitive.
    # re.DOTALL : Makes the '.' character match newlines as well. Crucial if the
    #             question or (more likely) the answer spans multiple lines.

    if not llm_output:
        print("`llm_output` should be defined.")
        return None

    pattern = r"Question:\s*(.*?)\s*Answer:\s*(.*)"
    match = re.search(pattern, llm_output, re.IGNORECASE | re.DOTALL)

    if match:
        question = match.group(1).strip()
        answer = match.group(2).strip()
        if not question or not answer:
            return None
        return {"question": question, "answer": answer}

    else:
        # Handle cases where the pattern wasn't found at all
        print(f"Warning: Could not parse Q/A structure from: {llm_output}")
        return None


def clean_text(text: str) -> str:
    """
    Cleans mathematical text by:
    1. Preserving LaTeX expressions like {\\displaystyle ...}
    2. Removing excessive newlines around variables
    3. Normalizing whitespace
    4. Keeping meaningful punctuation
    """
    # Remove invisible Unicode characters
    text = re.sub(r"[\u2060\u200b\u200c\u200d]", "", text)

    # Fix spacing around single variables
    text = re.sub(r"\n\s*([a-zA-Z])\s*\n", r" \1 ", text)

    # Normalize LaTeX displaystyle blocks
    text = re.sub(
        r"\{\s*\\displaystyle\s*([^}]+)\s*\}", r"{\\displaystyle \1}", text
    )

    # Compress multiple spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Fix spacing before punctuation
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)

    # Remove space after opening and before closing parentheses
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def split_text(text: str, max_size: int) -> list[str]:
    """Split text into chunks of maximum size, trying to break at sentence boundaries"""
    chunks = []
    current_chunk = ""

    # First try to split at paragraphs
    paragraphs = text.split("\n\n")

    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 > max_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para

    if current_chunk:
        chunks.append(current_chunk.strip())

    # If any chunk is still too large, split at sentence boundaries
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_size:
            final_chunks.append(chunk)
            continue

        sentences = re.split(r"(?<=[.!?])\s+", chunk)
        current_sentence_chunk = ""
        for sentence in sentences:
            if (
                len(current_sentence_chunk) + len(sentence) + 1 > max_size
                and current_sentence_chunk
            ):
                final_chunks.append(current_sentence_chunk.strip())
                current_sentence_chunk = sentence
            else:
                if current_sentence_chunk:
                    current_sentence_chunk += " " + sentence
                else:
                    current_sentence_chunk = sentence

        if current_sentence_chunk:
            final_chunks.append(current_sentence_chunk.strip())

    return final_chunks
