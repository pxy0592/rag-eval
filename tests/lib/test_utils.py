import pytest
from src.lib.utils import parse_qa_output


@pytest.mark.parametrize(
    "test_id, input_string, expected_output",
    [
        (
            "standard_case",
            "Question: What is the capital of France?\nAnswer: Paris.",
            {"question": "What is the capital of France?", "answer": "Paris."},
        ),
        (
            "different_casing",
            "question: who wrote hamlet?\nanswer: William Shakespeare",
            {"question": "who wrote hamlet?", "answer": "William Shakespeare"},
        ),
        (
            "extra_whitespace",
            "   Question:   How high is Mount Everest?  \n\n   Answer: \n 8,848 meters  ",
            {"question": "How high is Mount Everest?", "answer": "8,848 meters"},
        ),
        (
            "multi_line_answer",
            "Question: What are the primary colors?\nAnswer: The primary colors are:\n- Red\n- Yellow\n- Blue",
            {
                "question": "What are the primary colors?",
                "answer": "The primary colors are:\n- Red\n- Yellow\n- Blue",
            },
        ),
        (
            "multi_line_question_and_answer",
            "Question: List the first three planets\nfrom the sun.\nAnswer: The first three planets are:\n1. Mercury\n2. Venus\n3. Earth",
            {
                "question": "List the first three planets\nfrom the sun.",
                "answer": "The first three planets are:\n1. Mercury\n2. Venus\n3. Earth",
            },
        ),
        (
            "leading_text",
            "Here is the result:\nQuestion: What is 2+2?\nAnswer: 4.",
            {"question": "What is 2+2?", "answer": "4."},
        ),
        (
            "trailing_text",  # Note: Regex captures trailing text as part of the answer due to greedy '.*'
            "Question: What is H2O?\nAnswer: Water.\nThank you.",
            {"question": "What is H2O?", "answer": "Water.\nThank you."},
        ),
        (
            "embedded_text",
            "Some text before.\nQuestion: Test Question?\nAnswer: Test Answer.\nSome text after.",
            {"question": "Test Question?", "answer": "Test Answer.\nSome text after."},
        ),
        ("minimal_valid", "Question:Q\nAnswer:A", {"question": "Q", "answer": "A"}),
        (
            "chinese_full_width_labels",
            "问题：截至2014年，该单位共有多少人？\n答案：4797人。",
            {"question": "截至2014年，该单位共有多少人？", "answer": "4797人。"},
        ),
        (
            "json_code_block",
            "```json\n{\"question\": \"问题？\", \"answer\": \"答案。\"}\n```",
            {"question": "问题？", "answer": "答案。"},
        ),
    ],
)
def test_parse_qa_output_success(test_id, input_string, expected_output):
    """Tests various successful parsing scenarios."""
    print(f"Running test: {test_id}")  # Optional: print test id for easier debugging
    result = parse_qa_output(input_string)
    assert result == expected_output


# --- Test Cases for Failed Parsing (Should return None) ---
@pytest.mark.parametrize(
    "test_id, input_string",
    [
        ("empty_string", ""),
        ("none_input", None),  # Function should handle None input if type hints allow
        ("only_question_label", "Question: What is missing?"),
        ("only_answer_label", "Answer: This is an answer."),
        ("wrong_label_order", "Answer: A\nQuestion: Q"),
        ("missing_question_colon", "Question Whoops\nAnswer: Answer"),
        ("missing_answer_colon", "Question: Question\nAnswer Oops"),
        ("gibberish", "This string has neither label."),
        ("labels_no_content", "Question:\nAnswer:"),
        ("labels_whitespace_content", "Question:   \nAnswer: \t\n "),
        ("looks_like_qa_but_not", "This is a question. This is the answer."),
    ],
)
def test_parse_qa_output_failure(test_id, input_string):
    """Tests various scenarios where parsing should fail and return None."""
    # Handle the None input case specifically if needed, although the function handles it
    if input_string is None:
        assert parse_qa_output(input_string) is None
        return  # Skip rest of the test for None input

    print(f"Running test: {test_id}")  # Optional: print test id
    result = parse_qa_output(input_string)
    assert result is None


# --- Optional: Test specific edge cases if not covered by parametrize ---
# Example: Although covered above, you could write a dedicated test
def test_labels_with_only_whitespace_returns_none():
    """Ensures that labels followed only by whitespace results in None."""
    input_str = "Question: \t \n Answer:     "
    assert parse_qa_output(input_str) is None


def test_create_jsonl_file_writes_one_json_object_per_line(tmp_path):
    import json
    from pathlib import Path

    from src.lib.utils import create_jsonl_file

    path = Path(create_jsonl_file([{"question": "问题一"}, {"question": "问题二"}]))
    try:
        assert path.suffix == ".jsonl"
        assert [json.loads(line) for line in path.read_text().splitlines()] == [
            {"question": "问题一"},
            {"question": "问题二"},
        ]
    finally:
        path.unlink(missing_ok=True)
